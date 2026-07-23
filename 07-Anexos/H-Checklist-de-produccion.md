---
title: "Apéndice C - Checklist de producción"
aliases:
  - "Checklist de producción LLMOps"
  - "Apéndice C"
  - "Production checklist"
tags:
  - curso/llmops
  - apendice
  - checklist
  - despliegue
  - incidentes
  - rollback
parte: "Apéndices"
created: 2026-06-30
---


# Apéndice C - Checklist de producción

<!-- CURSO_NAV_TOP -->
[← Apéndice B - Patrones de diseño de sistemas](G-Patrones-de-diseno-de-sistemas.md) · [Índice](../README.md) · [Apéndice D - Guía de troubleshooting →](I-Troubleshooting-de-serving.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Capítulo avanzado
> Los conceptos se aplican a cualquier sistema. Los laboratorios de serving con CUDA se ejecutan mejor en WSL2/Linux o cloud; en Apple Silicon puedes practicar las ideas con llama.cpp, MLX o vLLM-Metal. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!abstract] Resumen
> Checklists accionables para llevar un servicio de inferencia de LLMs a producción y operarlo. Cubren cuatro fases: **pre-despliegue** (todo lo que debe estar verde antes de tocar producción), **go-live** (el corte controlado), **respuesta a incidentes** (cuando algo se rompe) y **rollback** (vuelta atrás segura). Marca cada casilla solo cuando exista evidencia, no por buena fe.

---

## Pre-despliegue

### Modelo y artefactos
- [ ] Versión del modelo (p. ej. Qwen3-0.6B) y *checksum* de los pesos fijados y registrados en el *model registry*.
- [ ] Configuración de cuantización documentada (esquema, bits, calibración) y validada frente a la línea base en calidad (ver [06 - Cuantización y compresión](../05-LLMOps/06-Cuantizacion-y-compresion-avanzada.md)).
- [ ] *Tokenizer* versionado junto al modelo; mismo `chat_template` que en evaluación.
- [ ] Artefactos de despliegue inmutables (imagen de contenedor con *digest* `@sha256`, no `latest`).

### Capacidad y rendimiento
- [ ] Memoria de KV cache dimensionada con la fórmula de [Apéndice A - Fundamentos matemáticos](F-Fundamentos-matematicos.md) para el contexto y batch objetivo.
- [ ] Pruebas de carga ejecutadas; **TTFT p50/p95**, **TPOT** (tiempo por token de salida) y **throughput** dentro de SLO.
- [ ] `max_model_len`, `max_num_seqs` y fracción de memoria GPU configurados y justificados.
- [ ] Comportamiento bajo saturación verificado (devuelve `429`, no se cuelga).

### Calidad y seguridad
- [ ] Suite de evaluación (*golden set*) pasa por encima del umbral acordado.
- [ ] Filtros de seguridad / moderación activos y probados con prompts adversariales.
- [ ] Límites de longitud de entrada y de generación aplicados (defensa frente a abuso de recursos).
- [ ] Secretos gestionados por *vault* / *managed identity*, nunca en variables ni imagen.

### Observabilidad
- [ ] Métricas expuestas (TTFT, TPOT, tokens/s, ocupación de KV cache, profundidad de cola) y *scrapeadas*.
- [ ] *Tracing* distribuido con spans de OpenTelemetry (ver [Apéndice E - Scaffold de implementación de referencia](J-Scaffold-de-implementacion.md)).
- [ ] Logs estructurados con `request_id` correlacionable.
- [ ] Dashboards y alertas creados **antes** del go-live (ver [11 - Observabilidad y monitorización](../05-LLMOps/10-Observabilidad-y-monitorizacion.md)).

### Infraestructura
- [ ] `/health/live` y `/health/ready` implementados y probados.
- [ ] Recursos (GPU, memoria, límites) declarados; sin *overcommit* accidental.
- [ ] Política de autoescalado definida con métrica correcta (cola/KV, no CPU%).
- [ ] Runbook de operación escrito y enlazado desde la alerta.

---

## Go-live

- [ ] Ventana de despliegue acordada y comunicada a los interesados.
- [ ] Despliegue **progresivo** (canary o blue-green): primer % de tráfico a la nueva versión.
- [ ] Métricas de la versión canary comparadas en vivo contra la estable (TTFT, error rate, calidad).
- [ ] *Smoke test* automático contra el endpoint de producción tras el corte (petición real end-to-end).
- [ ] `readiness` confirmada en todas las réplicas antes de subir el tráfico.
- [ ] Sin picos de errores 5xx ni de latencia durante el aumento gradual de tráfico.
- [ ] Criterio de éxito explícito (p. ej. "p95 TTFT < X ms y error rate < Y % durante 30 min").
- [ ] Persona responsable (*on-call*) identificada y disponible durante la ventana.
- [ ] Plan de rollback **revisado y a un clic de distancia** antes de empezar.

---

## Respuesta a incidentes

### Detección y triaje (primeros minutos)
- [ ] Confirmar el incidente con métricas, no por una sola alerta (descartar falso positivo).
- [ ] Declarar severidad (SEV1/2/3) y abrir canal de incidente.
- [ ] Asignar *Incident Commander*; nombrar quién comunica y quién investiga.

### Mitigación (estabilizar antes que entender)
- [ ] ¿Coincide en el tiempo con un despliegue reciente? Si sí, **rollback inmediato** (ver siguiente sección).
- [ ] Si hay saturación: activar autoescalado / añadir réplicas / activar *backpressure*.
- [ ] Si una réplica está degradada: drenarla del balanceador (readiness=false).
- [ ] Aplicar *feature flag* o degradación elegante (p. ej. bajar `max_tokens`, desactivar función costosa).

### Diagnóstico
- [ ] Consultar la [Apéndice D - Guía de troubleshooting](I-Troubleshooting-de-serving.md) por síntoma (TTFT alto, OOM de KV, etc.).
- [ ] Revisar logs estructurados filtrando por `request_id` de las peticiones afectadas.
- [ ] Capturar evidencia (métricas, trazas, dumps) **antes** de reiniciar nada.

### Cierre
- [ ] Confirmar recuperación con métricas durante un periodo de observación.
- [ ] Comunicar resolución a los interesados.
- [ ] Programar *post-mortem* sin culpa con acciones correctivas y dueños.

---

## Rollback

- [ ] Criterio de disparo de rollback definido **de antemano** (no improvisar bajo presión).
- [ ] Versión estable anterior (imagen con *digest* concreto) identificada y disponible.
- [ ] Compatibilidad verificada: el formato de la KV cache, el *tokenizer* y el `chat_template` de la versión anterior siguen siendo válidos.
- [ ] Migraciones de datos/configuración **reversibles** o con plan de compensación.
- [ ] Ejecutar rollback con el mismo mecanismo progresivo (canary inverso) cuando el tiempo lo permita.
- [ ] Drenar peticiones en vuelo de las réplicas nuevas antes de retirarlas (sin cortes bruscos).
- [ ] Confirmar `readiness` y métricas verdes en la versión restaurada.
- [ ] *Smoke test* post-rollback.
- [ ] Registrar el rollback (qué, cuándo, por qué) y enlazarlo al incidente.
- [ ] No re-desplegar la versión problemática hasta tener causa raíz y fix verificado.

> [!warning] Regla de oro
> Mitigar primero, entender después. Un rollback rápido casi siempre es mejor que un diagnóstico heroico con el servicio caído.

---

## Enlaces relacionados

- [10 - Despliegue en Azure ML](../05-LLMOps/09-Despliegue-en-Azure-ML.md)
- [11 - Observabilidad y monitorización](../05-LLMOps/10-Observabilidad-y-monitorizacion.md)
- [P3 - Proyecto - Sistema de serving en producción](../06-Proyectos/04-Sistema-de-serving-en-produccion.md)
- [Apéndice B - Patrones de diseño de sistemas](G-Patrones-de-diseno-de-sistemas.md)
- [Apéndice D - Guía de troubleshooting](I-Troubleshooting-de-serving.md)
- [Apéndice E - Scaffold de implementación de referencia](J-Scaffold-de-implementacion.md)

---

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Apéndice B - Patrones de diseño de sistemas](G-Patrones-de-diseno-de-sistemas.md) · [Índice](../README.md) · [Apéndice D - Guía de troubleshooting →](I-Troubleshooting-de-serving.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
