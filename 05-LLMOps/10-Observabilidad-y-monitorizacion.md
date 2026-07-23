---
title: "11 - Observabilidad y monitorización"
aliases:
  - Observabilidad LLMOps
  - Monitorización de modelos
  - Capítulo 11
tags:
  - curso/llmops
  - observabilidad
  - monitorizacion
  - mlflow
  - opentelemetry
  - azure-monitor
parte: "Parte 4 — Observabilidad y operaciones"
capitulo: 11
created: 2026-06-30
---


# Observabilidad y monitorización

<!-- CURSO_NAV_TOP -->
[← Despliegue en Azure ML](09-Despliegue-en-Azure-ML.md) · [Índice](../README.md) · [Optimización de costes →](11-Optimizacion-de-costes.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Capítulo avanzado
> Los conceptos se aplican a cualquier sistema. Los laboratorios de serving con CUDA se ejecutan mejor en WSL2/Linux o cloud; en Apple Silicon puedes practicar las ideas con llama.cpp, MLX o vLLM-Metal. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!abstract] En este capítulo
> Un sistema de inferencia que no se observa es una caja negra a la deriva. Aquí construimos la observabilidad de un servicio de LLM desde primeros principios: las **tres capas** que hay que vigilar, **MLflow** como plano de control, el **trazado por petición** con **OpenTelemetry**, la integración con **Azure Monitor** / **Application Insights**, los tres *dashboards* imprescindibles, las alertas que de verdad disparan (basadas en **SLO/SLI** y *burn rate*), la diferencia entre evaluación clásica y de **GenAI**, la detección de *drift* (deriva) y la disciplina de qué loguear y qué no. Todo anclado en el despliegue de **Qwen3-0.6B**.

## Las tres capas: infraestructura, servicio y modelo

La observabilidad de un LLM en producción no es una sola cosa. Son tres planos superpuestos, y confundirlos es la primera causa de *dashboards* inútiles.

| Capa | Pregunta que responde | Señales típicas |
|---|---|---|
| **Infraestructura** | ¿Está sana la máquina? | Uso de GPU/CPU, memoria VRAM, temperatura, I/O, red |
| **Servicio** | ¿Responde bien el endpoint? | Latencia (p50/p95/p99), throughput, tasa de error, profundidad de cola |
| **Modelo / calidad** | ¿Responde *bien de verdad*? | Tokens/s, longitud de salida, tasa de rechazo, métricas de calidad, *drift* |

La regla de primeros principios: **una métrica de capa superior nunca explica un fallo de capa inferior, pero sí al revés**. Si la latencia p99 (servicio) se dispara, la causa casi siempre está en infraestructura (VRAM saturada, *swapping* de KV-cache) o en el modelo (salidas más largas). Por eso el orden de diagnóstico es siempre de arriba hacia abajo en síntoma, y de abajo hacia arriba en causa.

> [!info] La distinción que más se olvida
> Un servicio puede tener latencia perfecta y *throughput* envidiable mientras genera basura. La capa de servicio y la capa de calidad son **ortogonales**. Necesitas ambas o no sabes nada.

## MLflow como plano de control

MLflow actúa como el **sistema de registro** de todo el ciclo de vida: experimentos, modelos versionados, métricas y artefactos. En LLMOps lo usamos como punto único de verdad sobre *qué modelo está dónde*.

```python
import mlflow

# Apuntamos al servidor de tracking (en Azure ML, la URI del workspace)
mlflow.set_tracking_uri("azureml://...")
mlflow.set_experiment("serving-qwen3-0.6b")

with mlflow.start_run(run_name="benchmark-baseline"):
    # Parámetros del despliegue que queremos poder reproducir
    mlflow.log_params({
        "modelo": "Qwen/Qwen3-0.6B",
        "precision": "bf16",
        "max_batch_tokens": 8192,
    })
    # Métricas observadas durante el benchmark de carga
    mlflow.log_metric("latencia_p95_ms", 312)
    mlflow.log_metric("tokens_por_segundo", 1840)
    # Artefacto: el informe completo de la prueba de carga
    mlflow.log_artifact("informe_carga.json")
```

El valor no está en loguear, sino en que **toda métrica de despliegue queda atada a una versión de modelo y a una configuración**. Cuando la latencia empeora tras un despliegue, MLflow te dice exactamente qué cambió.

## El *model registry* y los *aliases*

El **registro de modelos** (*model registry*) convierte un artefacto suelto en un objeto gobernable con versiones inmutables. La pieza moderna que sustituye a las antiguas "etapas" (*stages*) son los **alias**: punteros móviles a una versión concreta.

```python
from mlflow import MlflowClient

client = MlflowClient()

# Registramos una nueva versión del modelo
mv = client.create_model_version(
    name="qwen3-serving",
    source="runs:/<run_id>/model",
    run_id="<run_id>",
)

# El alias 'champion' apunta a la versión que sirve producción.
# 'challenger' apunta a la candidata que estamos evaluando en A/B.
client.set_registered_model_alias("qwen3-serving", "champion", version=mv.version)
```

El patrón **champion/challenger** desacopla el código del despliegue de la decisión de qué versión sirve. El servicio resuelve `models:/qwen3-serving@champion` en arranque; promover una nueva versión es mover el alias, no redesplegar. Esto habilita *rollbacks* en segundos.

```mermaid
graph LR
    V1[versión 3] -.->|alias retirado| X[ ]
    V2[versión 4] -->|@champion| PROD[Endpoint producción]
    V3[versión 5] -->|@challenger| AB[Tráfico A/B 5%]
    class PROD,AB internal-link;
```

## Trazado por petición con OpenTelemetry

Las métricas agregadas (p95, *throughput*) ocultan la cola de la distribución. Para entender *por qué* una petición concreta tardó 4 segundos necesitas **trazas distribuidas** (*distributed tracing*). **OpenTelemetry** (OTel) es el estándar vendor-neutral para esto: una petición genera un **span** (tramo) raíz y sub-*spans* anidados.

```python
from opentelemetry import trace

tracer = trace.get_tracer("serving.qwen3")

def generar(peticion):
    # Span raíz de toda la petición
    with tracer.start_as_current_span("inferencia") as span:
        # Los atributos son pares clave-valor que enriquecen la traza
        span.set_attribute("modelo.alias", "champion")
        span.set_attribute("tokens.entrada", peticion.n_tokens_in)

        with tracer.start_as_current_span("espera_en_cola") as s_cola:
            t_cola = encolar(peticion)
            s_cola.set_attribute("cola.profundidad", t_cola.depth)

        with tracer.start_as_current_span("prefill"):
            estado = prefill(peticion)  # procesado del prompt

        with tracer.start_as_current_span("decode") as s_dec:
            salida = decode(estado)
            s_dec.set_attribute("tokens.salida", salida.n_tokens)

        return salida
```

La granularidad correcta separa **prefill** (procesado del *prompt*, sensible a longitud de entrada) de **decode** (generación autoregresiva, sensible a longitud de salida). Sin esa separación no puedes saber si un pico de latencia viene de *prompts* largos o de respuestas largas. Los atributos clave (alias del modelo, tokens de entrada/salida, profundidad de cola) son los que luego permiten filtrar las trazas lentas.

> [!tip] No instrumentes a ciegas
> Un *span* sin atributos útiles es ruido. La pregunta antes de cada `set_attribute` es: *¿filtraría o agruparía una investigación por este valor?* Si no, no lo añadas.

## Azure Monitor y Application Insights

En Azure ML, las trazas y métricas fluyen hacia **Azure Monitor**, con **Application Insights** como el *backend* de telemetría de aplicación. El exportador OTLP de OpenTelemetry envía los *spans* allí, y se consultan con **KQL** (Kusto Query Language).

```kql
// Latencia p95 de decode por hora, últimas 6 horas
dependencies
| where name == "decode"
| where timestamp > ago(6h)
| summarize p95 = percentile(duration, 95) by bin(timestamp, 1h)
| render timechart
```

La integración no es solo visualización: Azure Monitor es también el motor de **alertas** sobre estas mismas series temporales, lo que cierra el círculo entre observar y reaccionar.

## Tres *dashboards* que todo equipo necesita

No hagas un *dashboard* gigante. Haz tres, cada uno para un público y un momento distinto.

> [!example] Los tres tableros
> 1. **Operacional (golden signals)** — para el ingeniero de guardia. Latencia p50/p95/p99, tasa de error, *throughput* (peticiones/s) y saturación (uso de GPU, profundidad de cola). Responde "¿está roto ahora?".
> 2. **De negocio / coste** — para el responsable de producto. Peticiones por usuario, tokens consumidos, **coste por petición**, distribución de longitud de salida. Responde "¿es sostenible?".
> 3. **De calidad del modelo** — para el equipo de ML. Tasa de rechazos, puntuaciones de evaluación *online*, señales de *drift*, distribución de la longitud de respuesta frente a la línea base. Responde "¿sigue siendo bueno?".

Las **golden signals** (latencia, tráfico, errores, saturación) provienen de la metodología SRE de Google y son el mínimo irrenunciable de la capa de servicio.

## Alertas que de verdad disparan: SLO/SLI y *burn rate*

Una alerta basada en un umbral fijo ("avisar si p95 > 500 ms") produce fatiga: salta de noche por un pico irrelevante y no salta cuando un error lento erosiona el servicio. La alternativa de primeros principios son los **SLO** (*Service Level Objectives*) sobre **SLI** (*Service Level Indicators*).

- **SLI**: una medida cruda de calidad de servicio. Ej.: proporción de peticiones servidas en < 800 ms.
- **SLO**: el objetivo sobre ese SLI. Ej.: 99 % de las peticiones en < 800 ms a 30 días.
- **Error budget** (presupuesto de error): el complemento. Con un SLO del 99 %, puedes "gastar" un 1 % de fallos.

La alerta correcta no mira el valor instantáneo, sino la **velocidad de consumo del presupuesto** (*burn rate*). Definimos el *burn rate* como:

$$
\text{burn rate} = \frac{\text{tasa de error observada}}{1 - \text{SLO}}
$$

Un *burn rate* de 1 significa que agotarás el presupuesto justo al final de la ventana. Un *burn rate* de 14,4 sostenido una hora consume el 2 % del presupuesto mensual: eso sí merece despertar a alguien.

> [!warning] Alertas multiventana
> La práctica recomendada combina una ventana corta (rápida, para incidentes agudos) y una larga (lenta, para degradaciones crónicas). Alertar solo cuando **ambas** superan el umbral elimina la mayoría de los falsos positivos.

## Evaluación: clásica frente a GenAI

La evaluación clásica de ML asume una **etiqueta de verdad** (*ground truth*) y métricas deterministas: *accuracy*, F1, AUC. Para un LLM generativo esto se rompe: no hay una única salida correcta y la métrica debe juzgar texto abierto.

| | Evaluación clásica | Evaluación GenAI |
|---|---|---|
| Verdad de referencia | Etiqueta única | Múltiples respuestas válidas |
| Métrica | Determinista (F1, AUC) | Semántica, a menudo con juez |
| Reproducibilidad | Total | Requiere fijar *seed*/temperatura |
| Coste por evaluación | Bajo | Alto (puede invocar otro LLM) |

La evaluación GenAI se trata en profundidad en [13 - Evaluación y monitorización de calidad](12-Evaluacion-y-calidad-en-produccion.md); aquí basta retener que la capa de calidad **necesita su propia infraestructura de evaluación**, distinta de las métricas de servicio.

## Detección de *drift* (deriva)

El **drift** es el cambio en la distribución de datos respecto a la línea base con la que se validó el modelo. En LLMs distinguimos:

- **Drift de entrada** (*data drift*): cambian los *prompts* que llegan (nuevos temas, otro idioma, *prompts* más largos).
- **Drift de salida**: cambia la distribución de respuestas (longitud media, tasa de rechazos) sin que cambie el modelo.
- **Drift de calidad** (*concept drift*): la respuesta correcta para un mismo *prompt* cambia con el tiempo (el mundo cambió).

La detección práctica compara distribuciones con tests estadísticos. Para una característica continua como la longitud del *prompt* se usa el estadístico de **Kolmogorov–Smirnov**, que mide la máxima diferencia entre dos funciones de distribución acumulada:

$$
D = \sup_x \left| F_{\text{ref}}(x) - F_{\text{actual}}(x) \right|
$$

Un $D$ grande con $p$ bajo señala que la distribución actual ya no se parece a la de referencia: hora de revisar el *eval set* y, posiblemente, reevaluar el modelo.

## Qué loguear y qué no: PII y coste

La tentación es loguear todo. Es un error operativo y legal.

> [!danger] No loguees *prompts* ni respuestas en crudo por defecto
> Pueden contener **PII** (información personal identificable). Almacenarlos sin control viola el RGPD y multiplica el coste de almacenamiento. Loguea **metadatos** (longitudes, latencias, alias del modelo, hashes), y solo el contenido bajo *opt-in* explícito, con anonimización y retención corta.

Reglas de primeros principios:

- **Cardinalidad bajo control**: nunca uses como etiqueta de métrica algo con cardinalidad ilimitada (ID de usuario, *prompt*). Hace explotar el coste del *backend* de métricas.
- **Muestreo de trazas**: en alto volumen, samplea (p. ej. 1–5 %) pero **garantiza el 100 % de las trazas de error**. El *tail-based sampling* hace exactamente eso.
- **Coste como métrica de primera clase**: cada petición logueada con sus tokens permite calcular el coste, base del [capítulo de costes](11-Optimizacion-de-costes.md).

> [!success] Puntos clave
> - La observabilidad tiene **tres capas** ortogonales: infraestructura, servicio y modelo/calidad; necesitas las tres.
> - **MLflow** es el plano de control; los **alias** (champion/challenger) desacoplan despliegue de decisión y habilitan *rollback* instantáneo.
> - **OpenTelemetry** da trazado por petición; separa siempre *prefill* de *decode*.
> - **Azure Monitor / Application Insights** unifican telemetría y alertas, consultables con KQL.
> - Construye **tres dashboards** (operacional, negocio/coste, calidad), no uno gigante.
> - Alerta por **SLO/SLI y burn rate multiventana**, no por umbrales fijos.
> - Distingue evaluación clásica de **GenAI** y vigila el **drift** con tests como KS.
> - **No loguees PII en crudo**; controla cardinalidad, samplea trazas y trata el coste como métrica de primera clase.

## Enlaces relacionados

- [10 - Despliegue en Azure ML](09-Despliegue-en-Azure-ML.md) — de dónde salen los endpoints que observamos
- [12 - Optimización de costes](11-Optimizacion-de-costes.md) — la telemetría de coste alimenta este capítulo
- [13 - Evaluación y monitorización de calidad](12-Evaluacion-y-calidad-en-produccion.md) — la capa de calidad en profundidad
- [05 - Batching y scheduling](05-Batching-y-scheduling.md) — origen de la profundidad de cola y la saturación
- [P3 - Proyecto - Sistema de serving en producción](../06-Proyectos/04-Sistema-de-serving-en-produccion.md) — donde se integra todo
- [Apéndice C - Checklist de producción](../07-Anexos/H-Checklist-de-produccion.md) — lista de verificación de observabilidad
- [Apéndice D - Guía de troubleshooting](../07-Anexos/I-Troubleshooting-de-serving.md) — diagnóstico de arriba abajo

---

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Despliegue en Azure ML](09-Despliegue-en-Azure-ML.md) · [Índice](../README.md) · [Optimización de costes →](11-Optimizacion-de-costes.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
