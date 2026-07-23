---
title: "Apéndice E - Scaffold de implementación de referencia"
aliases:
  - "Scaffold de referencia LLMOps"
  - "Apéndice E"
  - "Reference implementation scaffold"
tags:
  - curso/llmops
  - apendice
  - codigo
  - observabilidad
  - despliegue
  - patrones
parte: "Apéndices"
created: 2026-06-30
---


# Apéndice E - Scaffold de implementación de referencia

<!-- CURSO_NAV_TOP -->
[← Apéndice D - Guía de troubleshooting](I-Troubleshooting-de-serving.md) · [Índice](../README.md) · [Índice →](../README.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Capítulo avanzado
> Los conceptos se aplican a cualquier sistema. Los laboratorios de serving con CUDA se ejecutan mejor en WSL2/Linux o cloud; en Apple Silicon puedes practicar las ideas con llama.cpp, MLX o vLLM-Metal. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!abstract] Resumen
> Esqueleto de proyecto de calidad de producción para un servicio de inferencia de LLMs. Mostramos el **árbol de directorios** de referencia y, para cada patrón clave, un fragmento de Python listo para producción: **type hints**, **logging estructurado**, **spans de OpenTelemetry**, **retry boundaries**, **assets de despliegue** (Azure ML) y **utilidades de evaluación**. El servicio sirve un modelo tipo Qwen3-0.6B; las dimensiones de configuración están ancladas en él.

---

## Árbol de directorios del proyecto

```text
llmops-serving/
├── pyproject.toml              # deps, ruff, mypy, pytest
├── README.md
├── src/
│   └── serving/
│       ├── __init__.py
│       ├── config.py           # settings tipados (pydantic)
│       ├── logging_setup.py    # logging estructurado JSON
│       ├── telemetry.py        # inicialización de OpenTelemetry
│       ├── engine.py           # wrapper del motor de inferencia
│       ├── api.py              # endpoints HTTP + health checks
│       ├── retry.py            # retry boundaries (tenacity)
│       └── eval/
│           ├── golden_set.py   # casos de evaluación
│           └── metrics.py      # métricas de calidad
├── deploy/
│   ├── azureml/
│   │   ├── environment.yml     # entorno conda/imagen
│   │   ├── endpoint.yml        # managed online endpoint
│   │   └── deployment.yml      # deployment (instancia GPU)
│   └── Dockerfile
└── tests/
    ├── test_engine.py
    └── test_eval.py
```

---

## Patrón: configuración tipada y type hints

> [!note] Por qué
> Los *type hints* documentan contratos y permiten a `mypy` cazar errores antes de runtime. La configuración tipada con `pydantic` valida los parámetros de serving al arrancar (fallo rápido si una variable de entorno está mal).

```python
# src/serving/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuración del servicio, validada al arrancar."""
    model_id: str = "Qwen/Qwen3-0.6B"
    max_model_len: int = 4096          # longitud máxima de contexto
    max_num_seqs: int = 64             # secuencias concurrentes (batch)
    gpu_memory_fraction: float = 0.90  # fracción de VRAM para el motor
    dtype: str = "bfloat16"

    class Config:
        env_prefix = "SERVING_"        # SERVING_MAX_NUM_SEQS, etc.

settings = Settings()  # falla aquí si algún valor es inválido
```

---

## Patrón: logging estructurado

> [!note] Por qué
> Los logs en JSON son *parseables* por máquinas y correlacionables por `request_id`. Nunca uses `print`; nunca logues secretos ni el contenido íntegro del prompt en producción.

```python
# src/serving/logging_setup.py
import logging, json, sys
from typing import Any

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Adjunta campos extra (p. ej. request_id) inyectados vía `extra=`
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        return json.dumps(payload, ensure_ascii=False)

def configurar_logging(nivel: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=nivel, handlers=[handler])
```

---

## Patrón: spans de OpenTelemetry

> [!note] Por qué
> El *tracing* distribuido descompone el ciclo de vida de una petición (cola → prefill → decode → respuesta) en spans con atributos, permitiendo atribuir la latencia a cada fase. Es la base para diagnosticar TTFT/TPOT (ver [11 - Observabilidad y monitorización](../05-LLMOps/10-Observabilidad-y-monitorizacion.md)).

```python
# src/serving/telemetry.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def configurar_tracing(servicio: str = "llmops-serving") -> trace.Tracer:
    proveedor = TracerProvider()
    proveedor.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(proveedor)
    return trace.get_tracer(servicio)

tracer = configurar_tracing()

# Uso en el camino de inferencia:
def generar(prompt: str, request_id: str) -> str:
    with tracer.start_as_current_span("inferencia") as span:
        span.set_attribute("request_id", request_id)
        span.set_attribute("prompt.tokens", contar_tokens(prompt))
        with tracer.start_as_current_span("prefill"):
            ...  # procesar el prompt
        with tracer.start_as_current_span("decode"):
            ...  # generación autoregresiva
        return salida
```

---

## Patrón: retry boundaries

> [!note] Por qué
> Reintentar solo en los **límites** correctos (fallos transitorios e idempotentes) y con *backoff* exponponencial + *jitter* evita las *retry storms*. No reintentes errores de validación (4xx) ni operaciones no idempotentes.

```python
# src/serving/retry.py
from tenacity import (retry, stop_after_attempt, wait_exponential_jitter,
                      retry_if_exception_type)

class ErrorTransitorio(Exception):
    """Fallo recuperable (timeout, réplica reiniciando)."""

@retry(
    stop=stop_after_attempt(2),                          # 1 intento + 1 reintento
    wait=wait_exponential_jitter(initial=0.1, max=2.0),  # backoff + jitter
    retry=retry_if_exception_type(ErrorTransitorio),     # solo transitorios
    reraise=True,
)
def llamar_motor(peticion: dict) -> dict:
    """Frontera de reintento alrededor de una llamada idempotente al motor."""
    return motor.infer(peticion)
```

---

## Patrón: assets de despliegue (Azure ML)

> [!note] Por qué
> Infraestructura como código: el endpoint y el deployment son declarativos, versionables y reproducibles. La imagen se fija por *digest*, no por `latest` (ver [10 - Despliegue en Azure ML](../05-LLMOps/09-Despliegue-en-Azure-ML.md)).

```yaml
# deploy/azureml/deployment.yml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: qwen3-06b-blue
endpoint_name: llmops-serving
model: azureml:qwen3-0.6b:1          # modelo versionado en el registry
environment: azureml:serving-env:1
instance_type: Standard_NC4as_T4_v3  # instancia con GPU
instance_count: 1
liveness_probe:
  path: /health/live
  initial_delay: 10
readiness_probe:
  path: /health/ready              # falso hasta que el modelo carga
  initial_delay: 30
request_settings:
  max_concurrent_requests_per_instance: 64
```

```python
# src/serving/api.py — health checks que respaldan las probes
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/health/live")
def live() -> dict:
    return {"status": "alive"}      # ¿el proceso responde?

@app.get("/health/ready")
def ready(resp: Response) -> dict:
    if not engine.modelo_cargado() or engine.kv_saturada():
        resp.status_code = 503      # sácame del balanceador, no me reinicies
        return {"status": "not_ready"}
    return {"status": "ready"}
```

---

## Patrón: utilidades de evaluación

> [!note] Por qué
> Un *golden set* versionado y métricas reproducibles permiten detectar regresiones de calidad (ver [Apéndice D - Guía de troubleshooting](I-Troubleshooting-de-serving.md)) antes y después de cada despliegue, y comparar configuraciones de cuantización.

```python
# src/serving/eval/metrics.py
from dataclasses import dataclass

@dataclass
class CasoEval:
    prompt: str
    referencia: str

@dataclass
class ResultadoEval:
    exactitud: float        # fracción de aciertos sobre el golden set
    n_casos: int

def exact_match(prediccion: str, referencia: str) -> bool:
    """Métrica simple: coincidencia exacta tras normalizar espacios."""
    return prediccion.strip().lower() == referencia.strip().lower()

def evaluar(casos: list[CasoEval], generar) -> ResultadoEval:
    aciertos = sum(exact_match(generar(c.prompt), c.referencia) for c in casos)
    return ResultadoEval(exactitud=aciertos / len(casos), n_casos=len(casos))
```

> [!checklist] Antes de promover una build
> - [ ] `mypy src/` sin errores.
> - [ ] `pytest tests/` en verde.
> - [ ] `evaluar()` sobre el golden set por encima del umbral.
> - [ ] Imagen construida y publicada por *digest*.

---

## Enlaces relacionados

- [04 - El bucle de inferencia](../05-LLMOps/04-El-bucle-de-inferencia.md)
- [10 - Despliegue en Azure ML](../05-LLMOps/09-Despliegue-en-Azure-ML.md)
- [11 - Observabilidad y monitorización](../05-LLMOps/10-Observabilidad-y-monitorizacion.md)
- [P3 - Proyecto - Sistema de serving en producción](../06-Proyectos/04-Sistema-de-serving-en-produccion.md)
- [Apéndice C - Checklist de producción](H-Checklist-de-produccion.md)
- [Apéndice D - Guía de troubleshooting](I-Troubleshooting-de-serving.md)

---

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Apéndice D - Guía de troubleshooting](I-Troubleshooting-de-serving.md) · [Índice](../README.md) · [Índice →](../README.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
