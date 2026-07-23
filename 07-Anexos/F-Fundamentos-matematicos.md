---
title: "Apéndice A - Fundamentos matemáticos"
aliases:
  - "Fundamentos matemáticos LLMOps"
  - "Apéndice A"
  - "Math foundations"
tags:
  - curso/llmops
  - apendice
  - matematicas
  - atencion
  - cuantizacion
  - sampling
parte: "Apéndices"
created: 2026-06-30
---


# Apéndice A - Fundamentos matemáticos

<!-- CURSO_NAV_TOP -->
[← Troubleshooting: errores comunes y soluciones](E-Troubleshooting-local.md) · [Índice](../README.md) · [Apéndice B - Patrones de diseño de sistemas →](G-Patrones-de-diseno-de-sistemas.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Capítulo avanzado
> Los conceptos se aplican a cualquier sistema. Los laboratorios de serving con CUDA se ejecutan mejor en WSL2/Linux o cloud; en Apple Silicon puedes practicar las ideas con llama.cpp, MLX o vLLM-Metal. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!abstract] Resumen
> Este apéndice deriva, desde primeros principios, las matemáticas que sustentan el serving de LLMs. Empezamos por el **softmax numéricamente estable** (el ladrillo de toda la atención), seguimos con el **online softmax** que es el corazón de FlashAttention, formalizamos la **scaled dot-product attention**, cuantificamos la **memoria de la KV cache**, construimos un **modelo de throughput de decoding**, contamos los **parámetros de LoRA** y cerramos con las **derivaciones de sampling** (temperature y top-p). Todas las cifras concretas están ancladas en **Qwen3-0.6B** y sus dimensiones públicas; no inventamos números.

---

## Softmax numéricamente estable (truco de la resta del máximo)

El **softmax** convierte un vector de *logits* $\mathbf{z} \in \mathbb{R}^{V}$ en una distribución de probabilidad sobre el vocabulario $V$:

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.
$$

El problema es puramente aritmético: en `float32` el mayor valor representable antes de `inf` es $\approx 3.4\times10^{38}$, y $e^{89} \approx 4.5\times10^{38}$ ya desborda. Como los logits de un LLM tras varias capas pueden superar fácilmente $\pm 40$, el numerador $e^{z_i}$ produce `inf` o `NaN`.

La clave es que el softmax es **invariante a traslaciones**. Restando una constante $c$ a todos los logits:

$$
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} \, e^{-c}}{e^{-c}\sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

El factor $e^{-c}$ se cancela: el resultado es **idéntico**. Si elegimos $c = m = \max_j z_j$, entonces el mayor exponente es $z_i - m \le 0$, de modo que $e^{z_i - m} \in (0, 1]$ y **nunca desborda**. El sumando $e^{0}=1$ garantiza además que el denominador $\ge 1$, evitando la división por un número diminuto.

```python
import numpy as np

def softmax_estable(z: np.ndarray) -> np.ndarray:
    # Resta del máximo: invariante a traslaciones, evita overflow de exp().
    m = np.max(z)            # constante c = max(z)
    e = np.exp(z - m)        # cada término en (0, 1]
    return e / np.sum(e)     # denominador >= 1
```

> [!tip] LogSumExp
> La misma idea genera la identidad **log-sum-exp**, omnipresente en la log-verosimilitud:
> $$\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}.$$

---

## Online softmax: el corazón de FlashAttention

El softmax clásico necesita **dos pasadas** sobre los datos: una para hallar $m$ y otra para acumular la suma. FlashAttention no puede permitirse materializar la matriz de atención completa $S \in \mathbb{R}^{N\times N}$ en memoria, así que procesa la dimensión de clave en **bloques** y mantiene un softmax **incremental**.

Sea un flujo de valores $z_1, z_2, \dots$. Definimos dos estadísticos de estado tras ver $t$ elementos: el máximo corriente $m_t$ y la suma normalizada corriente $\ell_t$:

$$
m_t = \max(m_{t-1},\, z_t), \qquad
\ell_t = \ell_{t-1}\, e^{m_{t-1} - m_t} + e^{z_t - m_t}.
$$

La intuición del **factor de corrección** $e^{m_{t-1}-m_t}$: cuando aparece un valor mayor que el máximo previo, todos los términos acumulados con la base antigua $m_{t-1}$ deben re-escalarse a la nueva base $m_t$. Como $m_t \ge m_{t-1}$, el factor está en $(0,1]$ y nunca desborda.

Para la atención, además del escalar $\ell$ acumulamos el numerador vectorial $\mathbf{o}_t$ (la salida ponderada por los valores $\mathbf{v}$):

$$
\mathbf{o}_t = \mathbf{o}_{t-1}\, e^{m_{t-1} - m_t} + e^{z_t - m_t}\, \mathbf{v}_t.
$$

Al terminar, la salida correcta es $\mathbf{o}_T / \ell_T$. Esto demuestra que un softmax exacto puede computarse en **una sola pasada** con memoria $O(d)$ en lugar de $O(N)$, que es precisamente lo que permite a FlashAttention mantener todo en SRAM.

```python
def online_softmax_attention(q, K, V, bloque=64):
    # q: (d,), K: (N, d), V: (N, d). Procesa K/V por bloques sin materializar S.
    import numpy as np
    d = q.shape[0]
    m = -np.inf                 # máximo corriente
    l = 0.0                     # suma normalizada corriente
    o = np.zeros(d)             # numerador vectorial corriente
    escala = 1.0 / np.sqrt(d)
    for inicio in range(0, K.shape[0], bloque):
        Kb = K[inicio:inicio+bloque]
        Vb = V[inicio:inicio+bloque]
        s = (Kb @ q) * escala                 # logits del bloque
        m_nuevo = max(m, np.max(s))
        correccion = np.exp(m - m_nuevo)      # re-escala lo acumulado
        p = np.exp(s - m_nuevo)               # pesos del bloque
        l = l * correccion + np.sum(p)
        o = o * correccion + p @ Vb
        m = m_nuevo
    return o / l
```

> [!note] Por qué importa en LLMOps
> Sin materializar $S$, el coste de memoria de la atención pasa de $O(N^2)$ a $O(N)$. Para contextos largos esto es la diferencia entre caber o no caber en la GPU.

---

## Scaled dot-product attention

La operación central de un Transformer toma consultas $Q\in\mathbb{R}^{N\times d_k}$, claves $K\in\mathbb{R}^{N\times d_k}$ y valores $V\in\mathbb{R}^{N\times d_v}$:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.
$$

**¿Por qué el factor $1/\sqrt{d_k}$?** Si las componentes de $\mathbf{q}$ y $\mathbf{k}$ son independientes con media $0$ y varianza $1$, el producto escalar $\mathbf{q}\cdot\mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ tiene:

$$
\mathbb{E}[\mathbf{q}\cdot\mathbf{k}] = 0, \qquad
\text{Var}(\mathbf{q}\cdot\mathbf{k}) = \sum_{i=1}^{d_k}\text{Var}(q_i k_i) = d_k.
$$

La desviación típica crece como $\sqrt{d_k}$. Sin normalizar, para $d_k$ grande los logits se dispersan tanto que el softmax satura (gradientes casi nulos). Dividir por $\sqrt{d_k}$ devuelve la varianza a $\approx 1$, manteniendo el softmax en su régimen sensible.

En Qwen3-0.6B la dimensión por cabeza es $d_k = 128$ (con `hidden_size` 1024 y 16 cabezas de query; usa **GQA** con 8 cabezas de clave/valor), de modo que $\sqrt{d_k} = \sqrt{128} \approx 11.31$.

---

## Memoria de la KV cache

Durante el *decoding* autoregresivo se cachean las claves y valores de todos los tokens previos para no recomputarlos. La memoria total es:

$$
M_{\text{KV}} = 2 \cdot L \cdot H_{kv} \cdot d_k \cdot S \cdot B \cdot b,
$$

donde el $2$ cubre **K y V**, $L$ es el número de capas, $H_{kv}$ las cabezas de clave/valor (clave para GQA), $d_k$ la dimensión por cabeza, $S$ la longitud de secuencia, $B$ el tamaño de batch y $b$ los bytes por elemento (2 en `fp16`/`bf16`).

> [!example] Qwen3-0.6B, una secuencia de 4096 tokens en bf16
> Con $L=28$, $H_{kv}=8$, $d_k=128$, $S=4096$, $B=1$, $b=2$:
> $$M_{\text{KV}} = 2 \cdot 28 \cdot 8 \cdot 128 \cdot 4096 \cdot 1 \cdot 2 \approx 4.69\times10^{8}\ \text{bytes} \approx 0.44\ \text{GiB}.$$
> Observa el ahorro de GQA: con MHA ($H_{kv}=16$) la cifra se duplicaría a $\approx 0.88$ GiB.

La memoria escala **linealmente** con $S$ y $B$, lo que convierte a la KV cache en el cuello de botella dominante para batches grandes o contextos largos (ver [03 - Atención y KV cache](../05-LLMOps/03-Atencion-y-KV-cache.md)).

---

## Modelo de throughput de decoding

El decoding es **memory-bound**: en cada paso se leen todos los pesos del modelo desde HBM para generar un único token (por secuencia). El tiempo por token está acotado por el ancho de banda de memoria:

$$
t_{\text{token}} \approx \frac{P \cdot b}{\text{BW}_{\text{mem}}},
$$

donde $P$ es el número de parámetros, $b$ bytes por parámetro y $\text{BW}_{\text{mem}}$ el ancho de banda efectivo (GB/s).

> [!example] Qwen3-0.6B en bf16 sobre una GPU con BW = 1000 GB/s
> $P = 0.6\times10^{9}$, $b=2$:
> $$t_{\text{token}} \approx \frac{0.6\times10^{9}\cdot 2}{1000\times10^{9}} = 1.2\times10^{-3}\ \text{s} = 1.2\ \text{ms},$$
> es decir un techo teórico de $\approx 833$ tokens/s para un único stream.

El *batching* amortiza la lectura de pesos entre $B$ secuencias: con un batch de tamaño $B$ se leen los pesos una vez para producir $B$ tokens, así que el **throughput agregado** crece casi linealmente hasta saturar el cómputo (régimen compute-bound) o la memoria de la KV cache. Por eso el continuous batching ([05 - Batching y scheduling](../05-LLMOps/05-Batching-y-scheduling.md)) es la palanca número uno de throughput.

---

## Conteo de parámetros de LoRA

**LoRA** (Low-Rank Adaptation) congela la matriz de pesos original $W_0 \in \mathbb{R}^{d_{out}\times d_{in}}$ y aprende una actualización de **rango bajo**:

$$
W = W_0 + \Delta W = W_0 + B A, \qquad B\in\mathbb{R}^{d_{out}\times r},\ A\in\mathbb{R}^{r\times d_{in}},
$$

con $r \ll \min(d_{in}, d_{out})$. El número de parámetros entrenables **por matriz adaptada** es:

$$
P_{\text{LoRA}} = r\,(d_{in} + d_{out}),
$$

frente a $d_{in}\cdot d_{out}$ del fine-tuning completo. La salida se escala por $\alpha/r$ donde $\alpha$ es un hiperparámetro.

> [!example] Una proyección de Qwen3-0.6B con $d_{in}=d_{out}=1024$, $r=8$
> - Completo: $1024\times1024 = 1\,048\,576$ parámetros.
> - LoRA: $8\,(1024+1024) = 16\,384$ parámetros.
> - Ratio: $16\,384 / 1\,048\,576 \approx 1.56\%$.

Aplicando LoRA a las proyecciones de atención (q, k, v, o) de las 28 capas, los parámetros entrenables caen a un puñado de millones, lo que permite *fine-tuning* en una sola GPU pequeña (ver [09 - Fine-tuning y adaptación de dominio](../04-Adaptar/02-Fine-tuning-con-PEFT-y-QLoRA.md)).

---

## Derivaciones de sampling: temperature y top-p

### Temperature

El muestreo con **temperatura** $T>0$ reescala los logits antes del softmax:

$$
p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}.
$$

- $T \to 1$: distribución original.
- $T \to 0^+$: $p$ tiende a un *one-hot* sobre el argmax (greedy). Si $z_k$ es el máximo, $e^{z_i/T}/e^{z_k/T} = e^{(z_i - z_k)/T} \to 0$ para $i\neq k$.
- $T \to \infty$: distribución uniforme, porque $z_i/T \to 0$ y todos los exponentes se igualan.

La temperatura controla la **entropía** de la distribución: subirla aplana, bajarla afila.

### Top-p (nucleus sampling)

El **top-p** trunca la distribución al **núcleo** mínimo cuya masa acumulada alcanza el umbral $p$. Ordenando las probabilidades de mayor a menor, $p_{(1)}\ge p_{(2)}\ge\dots$, se elige el menor $k$ tal que:

$$
\sum_{i=1}^{k} p_{(i)} \ge p,
$$

se descartan el resto y se **renormaliza** sobre el núcleo $\mathcal{N}$:

$$
\tilde{p}_i = \frac{p_i}{\sum_{j\in\mathcal{N}} p_j}, \quad i\in\mathcal{N}.
$$

A diferencia del top-k (número fijo de candidatos), el top-p adapta el tamaño del núcleo al contexto: si una distribución está muy concentrada, el núcleo es pequeño; si es plana, se ensancha.

```python
import numpy as np

def sample_temp_topp(logits: np.ndarray, T: float = 0.8, p: float = 0.9) -> int:
    z = logits / T                                  # temperatura
    z -= np.max(z)                                  # estabilidad numérica
    probs = np.exp(z); probs /= probs.sum()         # softmax
    orden = np.argsort(probs)[::-1]                 # de mayor a menor
    acum = np.cumsum(probs[orden])
    k = np.searchsorted(acum, p) + 1                # menor núcleo con masa >= p
    nucleo = orden[:k]
    p_nucleo = probs[nucleo] / probs[nucleo].sum()  # renormalización
    return int(np.random.choice(nucleo, p=p_nucleo))
```

> [!warning] Orden de operaciones
> La temperatura se aplica **antes** del softmax y top-p **después**. Invertir el orden cambia la semántica: top-p sobre logits sin temperar usa una distribución distinta de la que finalmente se muestrea.

---

## Enlaces relacionados

- [03 - Atención y KV cache](../05-LLMOps/03-Atencion-y-KV-cache.md)
- [04 - El bucle de inferencia](../05-LLMOps/04-El-bucle-de-inferencia.md)
- [05 - Batching y scheduling](../05-LLMOps/05-Batching-y-scheduling.md)
- [06 - Cuantización y compresión](../05-LLMOps/06-Cuantizacion-y-compresion-avanzada.md)
- [09 - Fine-tuning y adaptación de dominio](../04-Adaptar/02-Fine-tuning-con-PEFT-y-QLoRA.md)
- [Apéndice B - Patrones de diseño de sistemas](G-Patrones-de-diseno-de-sistemas.md)
- [Apéndice E - Scaffold de implementación de referencia](J-Scaffold-de-implementacion.md)

---

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Troubleshooting: errores comunes y soluciones](E-Troubleshooting-local.md) · [Índice](../README.md) · [Apéndice B - Patrones de diseño de sistemas →](G-Patrones-de-diseno-de-sistemas.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
