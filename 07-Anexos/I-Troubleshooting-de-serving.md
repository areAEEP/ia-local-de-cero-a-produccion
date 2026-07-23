---
title: "Apéndice D - Guía de troubleshooting"
aliases:
  - "Troubleshooting LLMOps"
  - "Apéndice D"
  - "Guía de diagnóstico"
tags:
  - curso/llmops
  - apendice
  - troubleshooting
  - diagnostico
  - rendimiento
parte: "Apéndices"
created: 2026-06-30
---


# Apéndice D - Guía de troubleshooting

<!-- CURSO_NAV_TOP -->
[← Apéndice C - Checklist de producción](H-Checklist-de-produccion.md) · [Índice](../README.md) · [Apéndice E - Scaffold de implementación de referencia →](J-Scaffold-de-implementacion.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Capítulo avanzado
> Los conceptos se aplican a cualquier sistema. Los laboratorios de serving con CUDA se ejecutan mejor en WSL2/Linux o cloud; en Apple Silicon puedes practicar las ideas con llama.cpp, MLX o vLLM-Metal. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!abstract] Resumen
> Guía de diagnóstico orientada a **síntomas**. Para cada síntoma observable en un sistema de serving de LLMs damos la **causa probable**, cómo **diagnosticarlo** (con comandos y métricas concretas) y la **solución**. Úsala junto con el runbook y la [Apéndice C - Checklist de producción](H-Checklist-de-produccion.md) durante un incidente. Anclado en el serving de Qwen3-0.6B.

---

## Tabla maestra síntoma → causa → diagnóstico → solución

| Síntoma | Causa probable | Cómo diagnosticar | Solución |
|---|---|---|---|
| **TTFT alto** (time-to-first-token) | Cola de admisión saturada; prefill compitiendo con decode; prompts muy largos | Métrica de profundidad de cola; comparar TTFT con tamaño de prompt; `nvidia-smi dmon` | Activar **chunked prefill**; separar prefill/decode; subir réplicas; limitar longitud de prompt |
| **Baja utilización de GPU** | Batch pequeño; cuello de botella en CPU (tokenización, sampling); static batching | `nvidia-smi dmon -s u` (col. `sm`); perfilar CPU; revisar `max_num_seqs` | **Continuous batching**; subir `max_num_seqs`; mover tokenización fuera del *hot path* |
| **OOM de KV cache** | Demasiadas secuencias concurrentes o contexto demasiado largo para la memoria | Métrica de ocupación de KV; calcular con la fórmula de [Apéndice A - Fundamentos matemáticos](F-Fundamentos-matematicos.md) | Bajar `max_num_seqs`/`max_model_len`; subir fracción GPU; activar swapping/recompute; GQA/cuantización |
| **Throughput bajo** (tokens/s) | Batch efectivo bajo; precisión sin optimizar; kernels subóptimos | tokens/s agregado vs por secuencia; ¿está la GPU al 100 % SM? | Aumentar batch; usar bf16/cuantización; FlashAttention; perfilar kernels |
| **Regresiones de calidad** | Cambio de cuantización, `chat_template` o versión de modelo; *drift* de entrada | Re-ejecutar *golden set*; comparar salidas A/B vs línea base; revisar plantilla | Revertir cambio; recalibrar cuantización; fijar `chat_template`; ver [06 - Cuantización y compresión](../05-LLMOps/06-Cuantizacion-y-compresion-avanzada.md) |
| **OOM al cargar el modelo** | Pesos + activaciones no caben antes incluso de servir; fracción GPU mal puesta | Log de arranque (falla antes de `ready`); `nvidia-smi` durante la carga | Cuantizar pesos; bajar fracción de KV; usar GPU mayor; sharding/offload |
| **Latencia inter-token alta** (TPOT) | Decode memory-bound; batch demasiado grande saturando ancho de banda; *speculative decoding* desactivado | Medir TPOT p50/p95; modelo de throughput de [Apéndice A - Fundamentos matemáticos](F-Fundamentos-matematicos.md) | Ajustar batch al punto dulce; bf16/cuantización; considerar *speculative decoding* |
| **Fragmentación** (OOM con memoria "libre") | Asignación contigua de KV; bloques huérfanos; reservas máximas por secuencia | Memoria libre reportada > 0 pero falla la asignación; métrica de fragmentación | **PagedAttention** (bloques de tamaño fijo); reducir tamaño de bloque; reiniciar para defragmentar |

---

## Detalle por síntoma

### TTFT alto
> [!bug] Diagnóstico
> El TTFT mide cuánto tarda en llegar el **primer** token; lo domina el *prefill* (procesar el prompt completo) más el tiempo en cola. Si el TTFT escala con el tamaño del prompt, es prefill; si escala con la carga del sistema, es cola.

```bash
# Utilización y ancho de banda de memoria en vivo (intervalo 1 s)
nvidia-smi dmon -s um -d 1
# Métrica de servidor (ejemplo de endpoint Prometheus del servidor de inferencia)
curl -s localhost:8000/metrics | grep -E "time_to_first_token|num_requests_waiting"
```

**Solución:** el *chunked prefill* trocea prompts largos para que no acaparen una iteración entera, permitiendo intercalar decode de otras secuencias y reducir el TTFT de la cola (ver [05 - Batching y scheduling](../05-LLMOps/05-Batching-y-scheduling.md)).

### Baja utilización de GPU
> [!bug] Diagnóstico
> Si la columna `sm` de `nvidia-smi dmon` está baja mientras hay peticiones en cola, la GPU está hambrienta. El sospechoso habitual es la CPU (tokenización, *detokenization*, sampling en Python) o un batch demasiado pequeño.

```bash
nvidia-smi dmon -s u -d 1     # columna sm = % de ocupación de los SM
PID_OBJETIVO=1234             # sustituye 1234 por el PID real
py-spy top --pid "$PID_OBJETIVO"  # ¿se va el tiempo en CPU/Python?
```

**Solución:** activar continuous batching y subir `max_num_seqs`; sacar la tokenización del camino crítico.

### OOM de KV cache
> [!bug] Diagnóstico
> Ocurre en mitad de la generación, no al cargar. La ocupación de KV llega al 100 % y nuevas secuencias (o tokens de las existentes) no caben.

Aplica la fórmula $M_{\text{KV}} = 2\,L\,H_{kv}\,d_k\,S\,B\,b$ para saber cuántas secuencias concurrentes de longitud $S$ caben en tu presupuesto de memoria.

**Solución:** reducir `max_num_seqs` o `max_model_len`, activar swapping/recompute, o aprovechar GQA y cuantización del KV para reducir $b$.

### OOM al cargar el modelo
> [!bug] Diagnóstico
> Falla **antes** de pasar a `ready`. Distinto del OOM de KV: aquí ni siquiera caben pesos + buffer mínimo.

```bash
# Estimación rápida del tamaño de pesos: P * bytes
# Qwen3-0.6B en bf16: 0.6e9 * 2 ≈ 1.2 GB solo de pesos
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

**Solución:** cuantizar pesos, ajustar la fracción reservada para KV, o usar una GPU con más VRAM.

### Latencia inter-token alta (TPOT)
> [!bug] Diagnóstico
> El TPOT (time-per-output-token) mide la cadencia de generación tras el primer token. Es memory-bound: lo limita el ancho de banda al leer los pesos cada paso (ver el modelo de throughput de [Apéndice A - Fundamentos matemáticos](F-Fundamentos-matematicos.md)).

**Solución:** encontrar el tamaño de batch en el punto dulce (suficiente para amortizar lecturas sin saturar el ancho de banda), reducir bytes por parámetro con bf16/cuantización, y evaluar *speculative decoding*.

### Fragmentación
> [!bug] Diagnóstico
> Síntoma desconcertante: `nvidia-smi` reporta memoria libre pero la asignación falla, o caben menos secuencias de las que la fórmula predice. Causa: asignación contigua y reservas por el máximo.

**Solución:** PagedAttention elimina casi toda la fragmentación al usar bloques de tamaño fijo no contiguos (ver [Apéndice B - Patrones de diseño de sistemas](G-Patrones-de-diseno-de-sistemas.md)).

---

## Comandos y métricas de diagnóstico de referencia

> [!tip] Caja de herramientas
> ```bash
> # GPU: ocupación (u), memoria (m), reloj/energía (c/p)
> nvidia-smi dmon -s ucmp -d 1
> # Snapshot de memoria
> nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv -l 1
> # Métricas del servidor de inferencia (Prometheus)
> curl -s localhost:8000/metrics | grep -E "ttft|tpot|kv_cache_usage|num_requests"
> # Perfilado de CPU/Python en caliente
> py-spy top --pid <pid>
> # Trazas distribuidas: filtrar por request_id en el backend de OTel
> ```
> Las métricas clave a vigilar siempre: **TTFT p95**, **TPOT p95**, **tokens/s agregado**, **ocupación de KV cache** y **profundidad de cola** (ver [11 - Observabilidad y monitorización](../05-LLMOps/10-Observabilidad-y-monitorizacion.md)).

---

## Enlaces relacionados

- [03 - Atención y KV cache](../05-LLMOps/03-Atencion-y-KV-cache.md)
- [05 - Batching y scheduling](../05-LLMOps/05-Batching-y-scheduling.md)
- [06 - Cuantización y compresión](../05-LLMOps/06-Cuantizacion-y-compresion-avanzada.md)
- [11 - Observabilidad y monitorización](../05-LLMOps/10-Observabilidad-y-monitorizacion.md)
- [Apéndice A - Fundamentos matemáticos](F-Fundamentos-matematicos.md)
- [Apéndice B - Patrones de diseño de sistemas](G-Patrones-de-diseno-de-sistemas.md)
- [Apéndice C - Checklist de producción](H-Checklist-de-produccion.md)

---

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Apéndice C - Checklist de producción](H-Checklist-de-produccion.md) · [Índice](../README.md) · [Apéndice E - Scaffold de implementación de referencia →](J-Scaffold-de-implementacion.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
