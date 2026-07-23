---
title: Rutas y autoevaluación
curso: IA-Local-de-Cero-a-Produccion
modulo: "00.04"
---

# Elige tu itinerario y comprueba el aprendizaje

<!-- CURSO_NAV_TOP -->
[← Tu primera sesión de IA local](03-Tu-primera-IA-local.md) · [Índice](../README.md) · [Equivalencias entre Windows y macOS →](../PLATAFORMAS-Y-COMANDOS.md)
<!-- /CURSO_NAV_TOP -->



El curso es modular. La ruta correcta es la que te lleva a un resultado útil sin convertir tu hardware en una prueba de paciencia.

## Ruta esencial: de cero a una aplicación privada

Duración orientativa: 12-18 horas.

1. introducción completa;
2. qué es un LLM y arquitectura transformer;
3. inferencia y cuantización;
4. RAG local;
5. evaluación local;
6. proyecto de asistente local.

Puedes parar aquí con un resultado perfectamente válido.

### Prueba final

- el asistente responde con un modelo local;
- usa al menos tres documentos propios de prueba;
- cita qué fragmentos recuperó;
- no responde cuando no hay evidencia;
- tienes una tabla modelo/velocidad/memoria/calidad.

## Ruta creador: RAG, agentes, voz y visión

Duración orientativa: 20-30 horas.

Sigue la ruta esencial y añade:

1. agentes y MCP;
2. multimodal;
3. voz;
4. evaluación por componentes;
5. una interfaz o API local.

### Prueba final

Tu aplicación debe dejar claro cuándo el modelo:

- responde con conocimiento propio;
- usa documentos;
- llama una herramienta;
- procesa imagen o audio;
- falla y necesita confirmación humana.

## Ruta adaptación: cambia comportamiento, no "metas PDFs"

Duración orientativa: 20-40 horas, más tiempo de cómputo.

1. fundamentos, inferencia y evaluación;
2. decide RAG frente a fine-tuning;
3. MLX en Apple Silicon o PEFT/QLoRA en NVIDIA;
4. compara modelo base y adaptador;
5. explora merging, pruning y destilación solo después.

### Prueba final

No vale "parece mejor". Necesitas:

- dataset de entrenamiento y conjunto de evaluación separados;
- baseline antes de entrenar;
- criterios de aceptación;
- medición de regresiones;
- decisión final: desplegar, iterar o descartar.

## Ruta LLMOps: del modelo al sistema

Duración orientativa: 35-60 horas.

1. completa fundamentos e inferencia;
2. recorre LLMOps en orden;
3. construye el motor de inferencia;
4. ejecuta el proyecto de serving;
5. usa checklist, observabilidad y evaluación continua.

### Prueba final

Debes poder dibujar y defender:

```text
cliente → cola → scheduler → runtime → modelo/KV cache
                     ↓
          métricas, trazas y logs
                     ↓
            evaluación y alertas
```

Incluye objetivos de TTFT, ITL, throughput, memoria, calidad y coste.

## Rutas según plataforma

| Equipo | Ruta práctica | Evita al principio |
|---|---|---|
| Windows sin GPU dedicada | Ollama/LM Studio CPU, modelos 1B-4B, RAG | entrenamiento y modelos grandes |
| Windows + NVIDIA 8-16 GB | Ollama/llama.cpp CUDA; WSL2 para PEFT | multi-GPU y contexto extremo |
| Windows + NVIDIA 24 GB+ | QLoRA, vLLM en WSL2, serving | asumir que producción = abrir un puerto |
| Mac 8 GB | modelos 0,8B-2B, contexto corto | LoRA de 7B y multitarea pesada |
| Mac 16-32 GB | Ollama, llama.cpp Metal, MLX, 4B-14B Q4 | llenar toda la memoria unificada |
| Mac 64 GB+ | modelos 20B-32B Q4, LoRA y serving local | confundir memoria con throughput de datacenter |

## Semáforo de autoevaluación

Marca cada competencia:

- 🟢 puedo explicarla y repetirla sin copiar;
- 🟡 puedo hacerla con la guía;
- 🔴 aún no sé diagnosticarla.

| Competencia | Estado |
|---|---|
| Identifico RAM, VRAM y memoria unificada | |
| Elijo modelo y cuantización con margen | |
| Distingo GGUF, safetensors y runtime | |
| Levanto y cierro una API local | |
| Mido tokens/s y memoria | |
| Explico prefill, decode y KV cache | |
| Construyo y evalúo un RAG | |
| Sé cuándo no usar un agente | |
| Comparo base y LoRA con eval separada | |
| Defino métricas y rollback de producción | |

Si tienes un 🔴 en una competencia que el siguiente bloque da por sabida, vuelve al capítulo enlazado desde el [índice](../README.md). No pasa nada por saltar matemáticas en la primera vuelta; sí pasa por saltar evaluación.

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Tu primera sesión de IA local](03-Tu-primera-IA-local.md) · [Índice](../README.md) · [Equivalencias entre Windows y macOS →](../PLATAFORMAS-Y-COMANDOS.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
