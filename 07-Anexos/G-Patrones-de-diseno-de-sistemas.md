---
title: "Apéndice B - Patrones de diseño de sistemas"
aliases:
  - "Patrones de diseño LLMOps"
  - "Apéndice B"
  - "System design patterns"
tags:
  - curso/llmops
  - apendice
  - scheduling
  - memoria
  - tolerancia-fallos
  - balanceo-carga
parte: "Apéndices"
created: 2026-06-30
---


# Apéndice B - Patrones de diseño de sistemas

<!-- CURSO_NAV_TOP -->
[← Apéndice A - Fundamentos matemáticos](F-Fundamentos-matematicos.md) · [Índice](../README.md) · [Apéndice C - Checklist de producción →](H-Checklist-de-produccion.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Capítulo avanzado
> Los conceptos se aplican a cualquier sistema. Los laboratorios de serving con CUDA se ejecutan mejor en WSL2/Linux o cloud; en Apple Silicon puedes practicar las ideas con llama.cpp, MLX o vLLM-Metal. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!abstract] Resumen
> Catálogo de patrones de diseño para un sistema de serving de LLMs en producción. Para cada uno de los cuatro pilares —**scheduling**, **gestión de memoria**, **tolerancia a fallos** y **balanceo de carga**— planteamos el **problema**, el **patrón** que lo resuelve, los **trade-offs** que conlleva y un **pseudocódigo o diagrama** que lo concreta. Todo está anclado en el serving de modelos como Qwen3-0.6B.

---

## Algoritmos de scheduling

> [!question] Problema
> Las peticiones llegan de forma asíncrona, con prompts de longitud muy distinta y un número de tokens a generar desconocido a priori. El *static batching* (esperar a llenar un batch fijo) desperdicia GPU: las secuencias cortas terminan pronto pero quedan bloqueadas hasta que la más larga del batch acaba (efecto *head-of-line blocking*).

**Patrón: continuous batching (iteration-level scheduling).** En lugar de programar a nivel de petición, se programa a nivel de **iteración de decoding**. Tras cada paso, las secuencias que han emitido `<eos>` se retiran del batch y se admiten nuevas peticiones en su lugar, manteniendo la GPU siempre llena.

```text
bucle_scheduler():
    en_ejecucion = []          # secuencias activas
    cola_espera   = []         # peticiones pendientes
    mientras True:
        # 1) Admisión: rellenar huecos si hay presupuesto de KV cache
        mientras cola_espera y hay_presupuesto_kv() y len(en_ejecucion) < batch_max:
            en_ejecucion.append(cola_espera.pop_por_prioridad())
        # 2) Un paso de forward (prefill de nuevas + decode de activas)
        logits = modelo.step(en_ejecucion)
        # 3) Muestrear, anexar token, comprobar terminación
        para s en en_ejecucion:
            s.anexar(sample(logits[s]))
            si s.terminada():
                liberar_kv(s); responder(s); en_ejecucion.remove(s)
```

> [!info] Trade-offs
> - **A favor:** utilización de GPU muy alta, menor latencia media, throughput hasta varios × frente a static batching.
> - **En contra:** scheduler más complejo; mezclar *prefill* (compute-bound) y *decode* (memory-bound) en el mismo paso puede penalizar el TTFT. Mitigación: **chunked prefill** o separación prefill/decode (ver [05 - Batching y scheduling](../05-LLMOps/05-Batching-y-scheduling.md)).
> - **Política de prioridad:** FCFS es justo pero no optimiza colas; *shortest-job-first* reduce latencia media pero puede causar *starvation* de prompts largos.

---

## Gestión de memoria

> [!question] Problema
> La KV cache crece de forma impredecible (no se sabe cuántos tokens generará cada secuencia). Reservar el máximo por secuencia genera **fragmentación interna** masiva y limita el batch; reservar poco provoca OOM a mitad de generación.

**Patrón: paginación de la KV cache (PagedAttention).** Se inspira en la memoria virtual de los sistemas operativos. La KV cache se divide en **bloques** de tamaño fijo (p. ej. 16 tokens). Una **tabla de bloques** por secuencia mapea posiciones lógicas a bloques físicos, que no necesitan ser contiguos.

```text
┌──────────── Secuencia A (tabla de bloques) ─────────────┐
│ lógico 0 → físico 7  │ lógico 1 → físico 2 │ lógico 2 → físico 9 │
└─────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
   Pool físico de bloques (no contiguos, asignación bajo demanda)
   [b0][b1][b2*][...][b7*][b8][b9*]   * = en uso
```

> [!info] Trade-offs
> - **A favor:** fragmentación interna < 4 % (solo el último bloque parcial); permite *copy-on-write* para compartir prefijos comunes (prompts de sistema, *beam search*).
> - **En contra:** indirección extra en el kernel de atención (lectura de la tabla de bloques); requiere un kernel especializado tipo PagedAttention.
> - **Patrón complementario:** *prefix caching* —cachear el KV de prompts de sistema compartidos para no recomputar su prefill—. Y como válvula de seguridad, **swapping/recompute**: ante presión de memoria, expulsar bloques a CPU (swap) o descartarlos y recomputar el prefill (recompute) al re-admitir la secuencia.

---

## Tolerancia a fallos

> [!question] Problema
> Las GPUs fallan (ECC, caídas de driver, OOM transitorios), los pods se reinician y los modelos tardan decenas de segundos en cargar. Un fallo no debe tumbar el servicio ni perder peticiones en vuelo silenciosamente.

**Patrón: defensa en profundidad con health checks, reintentos idempotentes y circuit breaker.**

```text
cliente → [load balancer] → réplica
                              │
            ┌─────────────────┴──────────────────┐
            │ /health/live   (¿proceso vivo?)     │
            │ /health/ready  (¿modelo cargado?    │
            │                 ¿KV cache OK?)       │
            └──────────────────────────────────────┘
   fallo → readiness=false → LB retira la réplica → reintento en otra
```

- **Liveness vs readiness:** *liveness* reinicia el pod si el proceso está colgado; *readiness* lo saca del balanceo mientras carga el modelo o está saturado, **sin** reiniciarlo.
- **Reintentos idempotentes:** reintentar solo peticiones seguras y acotar con *budget* (p. ej. máx. 1 reintento) para no amplificar una caída en una *retry storm*.
- **Circuit breaker:** si una réplica supera un umbral de errores, se "abre" el circuito y se deja de enviarle tráfico durante un periodo, dándole tiempo a recuperarse.
- **Timeouts y backpressure:** límites de cola; si la cola está llena, devolver `429` rápido en lugar de aceptar trabajo que no se podrá atender.

> [!info] Trade-offs
> - **A favor:** degradación elegante; sin pérdida silenciosa de peticiones.
> - **En contra:** los reintentos mal acotados amplifican incidentes; readiness demasiado estricta reduce capacidad disponible. Hay que afinar umbrales con datos de [11 - Observabilidad y monitorización](../05-LLMOps/10-Observabilidad-y-monitorizacion.md).

---

## Balanceo de carga

> [!question] Problema
> El round-robin clásico asume peticiones homogéneas. En LLMs el coste varía órdenes de magnitud según la longitud del prompt y los tokens a generar, y el estado relevante (KV cache ocupada, longitud de cola) vive **en cada réplica**. Round-robin envía trabajo a réplicas ya saturadas.

**Patrón: balanceo consciente de la carga (least-loaded / power-of-two-choices) + afinidad de prefijo.**

```text
enrutar(peticion):
    # Power-of-two-choices: muestrear 2 réplicas al azar y elegir la menos cargada
    a, b = muestrear_dos(replicas)
    elegida = min(a, b, key=lambda r: r.peticiones_en_cola + r.tokens_kv_activos)
    # Afinidad de prefijo: si el prompt comparte prefijo cacheado, preferir esa réplica
    si peticion.hash_prefijo en cache_prefijos:
        elegida = cache_prefijos[peticion.hash_prefijo]
    enviar(elegida, peticion)
```

> [!info] Trade-offs
> - **Round-robin:** simple y sin estado, pero ignora la carga real.
> - **Least-loaded:** óptimo en teoría, pero requiere métricas frescas de cada réplica (coste de coordinación) y puede causar efecto "rebaño" si la información está obsoleta.
> - **Power-of-two-choices:** casi tan bueno como least-loaded con coste $O(1)$ de información; el patrón recomendado por defecto.
> - **Afinidad de prefijo:** maximiza el aprovechamiento del *prefix caching*, pero compite con el equilibrado puro de carga; hay que ponderar ambos objetivos.

> [!tip] Métrica de carga adecuada
> No uses CPU% ni RPS para balancear LLMs. Las señales correctas son la **profundidad de cola**, los **tokens de KV cache activos** y el **TTFT p95** por réplica.

---

## Enlaces relacionados

- [04 - El bucle de inferencia](../05-LLMOps/04-El-bucle-de-inferencia.md)
- [05 - Batching y scheduling](../05-LLMOps/05-Batching-y-scheduling.md)
- [10 - Despliegue en Azure ML](../05-LLMOps/09-Despliegue-en-Azure-ML.md)
- [11 - Observabilidad y monitorización](../05-LLMOps/10-Observabilidad-y-monitorizacion.md)
- [Apéndice A - Fundamentos matemáticos](F-Fundamentos-matematicos.md)
- [Apéndice C - Checklist de producción](H-Checklist-de-produccion.md)
- [Apéndice D - Guía de troubleshooting](I-Troubleshooting-de-serving.md)

---

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Apéndice A - Fundamentos matemáticos](F-Fundamentos-matematicos.md) · [Índice](../README.md) · [Apéndice C - Checklist de producción →](H-Checklist-de-produccion.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
