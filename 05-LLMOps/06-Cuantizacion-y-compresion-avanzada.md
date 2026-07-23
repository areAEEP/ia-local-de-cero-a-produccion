---
title: "06 - Cuantización y compresión"
aliases:
  - "Cuantización y compresión"
  - "Quantization"
  - "GPTQ AWQ INT8 INT4"
tags:
  - curso/llmops
  - cuantizacion
  - compresion
  - inferencia
  - memoria
  - precision-numerica
parte: 2
capitulo: 6
created: 2026-06-30
---

# Cuantización y compresión

<!-- CURSO_NAV_TOP -->
[← Batching y scheduling](05-Batching-y-scheduling.md) · [Índice](../README.md) · [Decodificación especulativa →](07-Decodificacion-especulativa.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Capítulo avanzado
> Los conceptos se aplican a cualquier sistema. Los laboratorios de serving con CUDA se ejecutan mejor en WSL2/Linux o cloud; en Apple Silicon puedes practicar las ideas con llama.cpp, MLX o vLLM-Metal. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!abstract] En este capítulo
> La cuantización (*quantization*) reduce la precisión numérica con la que representamos un modelo: de 32 o 16 bits por número a 8 o incluso 4 bits. El premio es doble: **menos memoria** (los pesos ocupan una fracción) y **más velocidad** (mover menos bytes acelera kernels memory-bound, ver [05 - Batching y scheduling](05-Batching-y-scheduling.md)). El reto es no destruir la calidad. Recorreremos los **formatos numéricos** y su anatomía de bits; **dónde** se cuantiza (pesos, activaciones, KV cache); la diferencia entre **PTQ** y **QAT**; los métodos **GPTQ** y **AWQ**; implementaremos **un cuantizador INT8 simétrico desde cero**; veremos cómo **medir la pérdida de calidad**; y revisaremos el soporte en motores de producción. Anclamos en **Qwen3-0.6B**.

## Formatos numéricos

Un número en coma flotante se compone de tres campos: **signo** (1 bit), **exponente** (gobierna el *rango*, los órdenes de magnitud representables) y **mantisa** o *significand* (gobierna la *precisión*, cuántos valores distintos hay dentro de cada orden de magnitud). Un número entero (INT) reparte sus bits de otra forma: no tiene exponente, solo magnitud.

| Formato | Bits totales | Signo | Exponente | Mantisa | Característica |
|---|---|---|---|---|---|
| **FP32** | 32 | 1 | 8 | 23 | Precisión completa; referencia de entrenamiento |
| **FP16** | 16 | 1 | 5 | 10 | Mucha precisión pero **rango estrecho** (riesgo de overflow) |
| **BF16** | 16 | 1 | 8 | 7 | Mismo rango que FP32, menos precisión; estándar de entrenamiento moderno |
| **FP8** (E4M3) | 8 | 1 | 4 | 3 | Coma flotante en 8 bits; rango razonable, poca precisión |
| **INT8** | 8 | (1) | — | — | Entero; 256 niveles; necesita escala |
| **INT4** | 4 | (1) | — | — | Entero; 16 niveles; agresivo, requiere métodos cuidadosos |

> [!info] BF16 frente a FP16: por qué importa el reparto de bits
> Ambos ocupan 16 bits, pero **BF16** conserva los 8 bits de exponente de FP32, así que cubre el mismo rango dinámico (evita overflow/underflow en entrenamiento) sacrificando mantisa. **FP16** invierte el compromiso: más mantisa (precisión), menos exponente (rango). Para inferencia con pesos bien acotados, FP16 suele bastar; para entrenamiento estable, BF16 es preferible.

La idea central de la cuantización entera: representamos un rango continuo de valores reales con un número finito de niveles enteros mediante una **escala** $s$ (y opcionalmente un *zero-point* $z$):

$$
x_{\text{real}} \approx s \cdot (q - z), \qquad q \in \mathbb{Z}
$$

Pasar de FP16 (16 bits) a INT4 (4 bits) divide la memoria de pesos **por cuatro**. Para Qwen3-0.6B eso significa pasar de ~1,2 GB a ~0,3 GB, liberando VRAM para KV cache y batches mayores.

## Dónde se aplica la cuantización

No todo se cuantiza igual ni con la misma facilidad:

- **Pesos (*weights*).** Los más fáciles y rentables: son **estáticos** (no cambian en inferencia), así que se cuantizan una vez offline. Es el objetivo principal (W8, W4). Gran ahorro de memoria y de ancho de banda.
- **Activaciones (*activations*).** Cambian con cada entrada y contienen **outliers** (valores atípicos enormes) que dificultan la cuantización. La cuantización de activaciones (esquemas W8A8) acelera el cómputo usando aritmética entera, pero es más delicada.
- **KV cache.** En contextos largos la KV cache (ver [03 - Atención y KV cache](03-Atencion-y-KV-cache.md)) domina la memoria. Cuantizarla a FP8 o INT8 permite **contextos más largos** o **más concurrencia**, a costa de algo de calidad en la atención.

> [!warning] El problema de los outliers en activaciones
> En modelos grandes, unas pocas dimensiones de las activaciones presentan magnitudes muchísimo mayores que el resto. Si aplicas una sola escala global, esos outliers obligan a una escala enorme que aplasta la resolución del resto de valores. Por eso la cuantización de activaciones necesita escalas por canal/grupo o técnicas que reubiquen la dificultad (es justo el problema que ataca AWQ).

## Calibración: PTQ frente a QAT

Cuantizar bien requiere elegir buenas escalas, y para eso hay dos filosofías:

**PTQ (*Post-Training Quantization*).** Se cuantiza un modelo **ya entrenado**, sin reentrenar. Se pasa un pequeño **conjunto de calibración** (unas decenas o centenares de muestras representativas) por el modelo para observar los rangos de activaciones y elegir escalas. Es rápido, barato y no necesita el dataset original ni la infraestructura de entrenamiento. Es lo habitual en LLMOps. GPTQ y AWQ son métodos PTQ.

**QAT (*Quantization-Aware Training*).** Se **simula** la cuantización **durante el entrenamiento** (o un fine-tuning), de modo que el modelo aprende pesos robustos a la pérdida de precisión. Da la mejor calidad a bits muy bajos, pero exige reentrenar: caro, lento y muchas veces inviable para modelos de terceros.

> [!tip] Regla práctica
> Empieza siempre por **PTQ**: para W8 (8 bits) la pérdida de calidad suele ser despreciable y el ahorro inmediato. Reserva **QAT** para cuando necesites bits muy agresivos (INT4 o menos) en un modelo crítico y tengas presupuesto de entrenamiento. El conjunto de calibración debe parecerse a tu tráfico real; uno mal elegido genera escalas malas.

## GPTQ

**GPTQ** (*Generative Pre-trained Transformer Quantization*) es un método PTQ que cuantiza los pesos **capa a capa**, minimizando el error que la cuantización introduce **en la salida de cada capa**, no en los pesos en sí. La diferencia es clave: no buscamos que $\hat{W} \approx W$, sino que $\hat{W}x \approx Wx$ para las entradas $x$ vistas en calibración.

Plantea, por cada capa lineal, un problema de mínimos cuadrados:

$$
\arg\min_{\hat{W}} \; \big\| W X - \hat{W} X \big\|_2^2
$$

donde $X$ son las activaciones de calibración. GPTQ cuantiza los pesos **uno a uno** y, tras fijar cada peso a su valor cuantizado, **ajusta los pesos aún no cuantizados** de esa capa para **compensar** el error introducido, usando información de segundo orden (una aproximación del Hessiano). Permite bajar a **INT4** con una pérdida de calidad contenida. Es un método **centrado en el peso**.

## AWQ

**AWQ** (*Activation-aware Weight Quantization*) parte de una observación: **no todos los pesos son igual de importantes**. Un pequeño porcentaje de canales de pesos (los que se multiplican por las activaciones de mayor magnitud) son **salientes** (*salient*) y concentran casi todo el impacto en la calidad.

En lugar de protegerlos guardándolos en alta precisión (lo que rompería el formato uniforme y el hardware), AWQ aplica un **reescalado por canal**: multiplica los canales salientes por un factor $s$ antes de cuantizar y divide por $s$ la activación correspondiente, dejando el producto matemáticamente equivalente pero **moviendo la dificultad** fuera de los pesos críticos. El factor de escala se busca minimizando el error de salida sobre el conjunto de calibración.

$$
W x = (W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} x)
$$

> [!info] GPTQ vs. AWQ en una frase
> **GPTQ** corrige el error iterativamente compensando con los pesos restantes (centrado en el peso). **AWQ** protege los canales importantes reescalándolos antes de cuantizar (centrado en la activación). Ambos son PTQ de pesos para INT4 y, en la práctica, dan calidades comparables; el ganador depende del modelo y del kernel disponible en tu motor.

## Un cuantizador INT8 simétrico desde cero

Vamos a construir la cuantización **simétrica** (zero-point = 0), la más común para pesos. "Simétrica" significa que el rango de valores se centra en cero: $[-\alpha, +\alpha]$. Con 8 bits con signo, el rango entero es $[-127, 127]$ (reservamos $-128$ para conservar simetría).

La escala se calcula a partir del valor absoluto máximo del tensor:

$$
s = \frac{\max |x|}{127}, \qquad q = \text{clamp}\big(\text{round}(x / s),\, -127,\, 127\big), \qquad \hat{x} = s \cdot q
$$

```python
import numpy as np

def cuantizar_int8_simetrico(x: np.ndarray):
    """Cuantiza un tensor float a INT8 simétrico (zero-point = 0).

    Devuelve los enteros int8 y la escala necesaria para deshacer
    la cuantizacion. Simetrico => el cero real se mapea al cero entero.
    """
    # 1) Hallar el valor absoluto maximo: define el rango [-amax, +amax].
    amax = np.max(np.abs(x))
    # Proteccion: si el tensor es todo ceros, evitamos dividir por cero.
    if amax == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0

    # 2) Escala: repartimos el rango real entre los 127 niveles positivos.
    #    Usamos 127 (no 128) para mantener simetria perfecta en torno a 0.
    escala = amax / 127.0

    # 3) Cuantizar: dividir por la escala, redondear al entero mas cercano
    #    y recortar (clamp) al rango representable [-127, 127].
    q = np.round(x / escala)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, escala


def descuantizar_int8(q: np.ndarray, escala: float) -> np.ndarray:
    """Reconstruye el float aproximado: x_hat = escala * q."""
    return q.astype(np.float32) * escala


# --- Ejemplo de uso ---
pesos = np.array([0.12, -0.45, 0.98, -1.30, 0.03], dtype=np.float32)
q, s = cuantizar_int8_simetrico(pesos)
recuperado = descuantizar_int8(q, s)

print("Escala:", s)                 # amax (1.30) / 127
print("Enteros int8:", q)           # valores en [-127, 127]
print("Recuperado:", recuperado)    # aproximacion del original
# El error de cuantizacion por elemento esta acotado por escala/2.
error_max = np.max(np.abs(pesos - recuperado))
print("Error maximo:", error_max)   # <= escala / 2
```

> [!example] Qué observar en el resultado
> El **error de cuantización** de cada elemento está acotado por $s/2$ (la mitad de un paso de escala). Por eso un único outlier grande (que infla `amax` y por tanto `escala`) degrada la resolución de **todos** los demás valores: es la raíz del problema de outliers y la motivación de cuantizar **por grupos** (un `amax`/escala por cada bloque de, p. ej., 128 pesos) en lugar de uno global por tensor.

> [!tip] Cuantización por grupos (*group-wise*)
> Los métodos de producción no usan una escala única por tensor sino una **escala por grupo** (grupos de 64 o 128 elementos). Así un outlier solo afecta a su grupo. Es el motivo por el que GPTQ/AWQ a INT4 funcionan razonablemente: combinan group-wise con su tratamiento específico del error.

## Medir la pérdida de calidad

Cuantizar sin medir es ir a ciegas. Hay dos familias de métricas:

**Perplejidad (*perplexity*, PPL).** Mide lo "sorprendido" que está el modelo ante un texto de referencia; es la exponencial de la *cross-entropy* media:

$$
\text{PPL} = \exp\!\left(-\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i \mid x_{<i})\right)
$$

Más baja es mejor. Es rápida de calcular sobre un corpus fijo (p. ej. WikiText) y muy útil para **comparar** el modelo cuantizado contra el original: un aumento pequeño de PPL indica degradación leve.

> [!warning] La perplejidad no lo cuenta todo
> Un modelo puede mantener una PPL casi idéntica y aun así fallar más en tareas concretas (razonamiento, código, seguimiento de instrucciones). La PPL es un **proxy barato**, no la verdad final.

**Métricas downstream (de tarea).** Evalúan el comportamiento en lo que de verdad importa: precisión (*accuracy*) en benchmarks de conocimiento o razonamiento, tasa de aciertos en código, calidad de respuestas con un juez, etc. Son más caras pero capturan degradaciones que la PPL no ve.

> [!tip] Protocolo mínimo de validación
> 1. Mide la **PPL** del modelo original y del cuantizado sobre el mismo corpus. Un delta pequeño es buena señal de partida.
> 2. Ejecuta **al menos una métrica downstream** representativa de tu caso de uso real.
> 3. Compara también **latencia y throughput** (ver [05 - Batching y scheduling](05-Batching-y-scheduling.md)) para confirmar que el ahorro de memoria se traduce en velocidad, no solo en VRAM libre.
> No reportes cifras sin medirlas en **tu** modelo y **tu** tarea: las ganancias y pérdidas varían mucho entre familias de modelos.

## Soporte de cuantización en motores de producción

No basta con cuantizar el fichero de pesos; el **motor de inferencia** necesita **kernels** que sepan ejecutar la aritmética cuantizada eficientemente.

**vLLM.** Soporta cargar modelos cuantizados en formatos populares (GPTQ, AWQ) y esquemas como W8A8/FP8 e INT8/INT4, además de cuantización de la KV cache (FP8). Detecta el formato en la configuración del modelo y selecciona los kernels adecuados. Es la vía más directa para servir un Qwen3-0.6B cuantizado con continuous batching.

**TensorRT-LLM** (NVIDIA). Compila el modelo en *engines* optimizados para una GPU concreta, con soporte fuerte de **FP8** e **INT4/INT8** y fusión de kernels muy agresiva. Suele dar el máximo rendimiento en hardware NVIDIA reciente, a costa de un paso de compilación y menos portabilidad.

> [!danger] El formato debe casar con el kernel
> Un checkpoint cuantizado con un método X solo corre rápido si el motor tiene el kernel para X en tu GPU. Un formato sin kernel optimizado puede **dequantizar al vuelo a FP16** y perder toda la ventaja de velocidad (conservando solo el ahorro de memoria). Verifica siempre la compatibilidad formato–motor–GPU antes de comprometerte con un esquema.

> [!success] Puntos clave
> - Los **formatos** reparten bits entre rango (exponente) y precisión (mantisa); INT8/INT4 ahorran memoria proporcionalmente a los bits, pero necesitan **escala**.
> - Se cuantizan sobre todo los **pesos** (estáticos, fáciles); las **activaciones** (outliers) y la **KV cache** son más delicadas.
> - **PTQ** (sin reentrenar, con calibración) es la opción por defecto; **QAT** reserva su coste para bits muy agresivos.
> - **GPTQ** compensa el error con los pesos restantes; **AWQ** reescala los canales salientes antes de cuantizar; ambos llegan a INT4 con pérdida contenida.
> - Un **cuantizador INT8 simétrico** se reduce a escala = $\max|x|/127$, round y clamp; el error por elemento está acotado por $s/2$, lo que motiva la cuantización **por grupos**.
> - **Mide** con perplejidad y métricas downstream; no fíes solo en la PPL.
> - El **motor** (vLLM, TensorRT-LLM) debe tener el **kernel** del formato elegido en tu GPU, o pierdes la ventaja de velocidad.

## Enlaces relacionados
- [05 - Batching y scheduling](05-Batching-y-scheduling.md) — menos bytes por peso acelera kernels memory-bound y libera VRAM para batches mayores.
- [03 - Atención y KV cache](03-Atencion-y-KV-cache.md) — cuantizar la KV cache amplía el contexto y la concurrencia.
- [07 - Decodificación especulativa](07-Decodificacion-especulativa.md) — otra palanca de aceleración, combinable con cuantización.
- [08 - De una GPU a inferencia multi-GPU](08-De-una-GPU-a-multi-GPU.md) — cuantizar puede evitar tener que repartir un modelo entre GPU.
- [11 - Observabilidad y monitorización](10-Observabilidad-y-monitorizacion.md) — vigilar calidad y latencia tras cuantizar.
- [12 - Optimización de costes](11-Optimizacion-de-costes.md) — la cuantización es una de las palancas de coste por token más directas.
- [Apéndice B - Patrones de diseño de sistemas](../07-Anexos/G-Patrones-de-diseno-de-sistemas.md) — gestión de variantes de modelo y despliegue.

---

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Batching y scheduling](05-Batching-y-scheduling.md) · [Índice](../README.md) · [Decodificación especulativa →](07-Decodificacion-especulativa.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
