---
tags:
  - curso/ia-local
  - modelos
  - apple-silicon
  - 24gb
curso: IA-Local-de-Cero-a-Produccion
modulo: anexo-modelos
estado: completo
last_review: 2026-07-05
---

# Modelos recomendados para Mac M2 24 GB

<!-- CURSO_NAV_TOP -->
[← Chuleta de comandos](C-Chuleta-de-comandos-original-Mac.md) · [Índice](../README.md) · [Troubleshooting: errores comunes y soluciones →](E-Troubleshooting-local.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Ruta Apple Silicon
> Este capítulo nació para Mac con Apple Silicon. Si usas Windows, quédate con los conceptos y sigue la alternativa indicada en [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!goals] Objetivos de aprendizaje
> - Elegir modelos que realmente caben en 24 GB.
> - Comparar usos: chat, código, razonamiento, fine-tuning y agentes.
> - Mantener una tabla viva de tamaños y rendimiento medidos.


## Regla práctica

Para uso diario, prioriza modelos 3B-8B en Q4/Q5. Para calidad extra, prueba 14B Q4 con contexto moderado. Si necesitas 30B+, usa cloud.

Relacionado: [01-Fundamentos](../01-Fundamentos/01-Que-es-un-LLM.md), [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md), [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

## Tabla base

Los tamaños son orientativos. Mide en tu máquina.

| Modelo | Uso | Quant recomendada | RAM aprox. | Comentario |
|---|---|---:|---:|---|
| Qwen2.5 3B Instruct | chat rápido, agentes | Q4/MLX 4-bit | 3-6 GB total | ideal para iterar |
| Qwen2.5 7B Instruct | general técnico | Q4_K_M o MLX 4-bit | 10-14 GB total | recomendación principal |
| Qwen2.5 14B Instruct | más calidad | Q4_K_M | 16-22 GB total | viable, vigilar contexto |
| Llama 3.1 8B Instruct | general | Q4_K_M/Q5_K_M | 11-15 GB total | muy equilibrado |
| Mistral 7B Instruct | general/código ligero | Q4_K_M | 10-14 GB total | rápido y probado |
| Mistral Nemo 12B | general | Q4_K_M | 15-21 GB total | justo pero viable |
| Gemma 2 9B | razonamiento ligero | Q4_K_M | 12-17 GB total | revisar licencia |
| Phi mini | rápido/coste bajo | Q4/Q8 | 4-8 GB total | útil para agentes |
| DeepSeek destilado 7B/8B | razonamiento | Q4_K_M | 10-15 GB total | bueno para pruebas |

## Recomendaciones por caso

### Asistente general local

- Qwen2.5 7B Instruct Q4_K_M.
- Llama 3.1 8B Instruct Q4_K_M.

### Agentes con muchas llamadas

- Qwen2.5 3B Instruct 4-bit.
- Phi mini cuantizado.

Motivo: los agentes hacen muchas iteraciones. Velocidad > máxima calidad.

### Código

- Qwen2.5 Coder 7B si está disponible en tu herramienta.
- DeepSeek Coder destilado pequeño si cabe.

### Fine-tuning local

- Qwen2.5 3B Instruct 4-bit para aprender.
- Qwen2.5 7B Instruct 4-bit para proyecto final.

Relacionado: [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

### Contexto largo

En 24 GB, contexto largo es el enemigo silencioso. Mejor:

- modelo 7B Q4;
- contexto 4k-8k;
- RAG o recuperación externa;
- no intentar meter documentos enteros sin necesidad.

## Plantilla para mediciones propias

| Fecha | Modelo | Herramienta | Quant | Contexto | Prompt | Tokens/s | Memoria | Nota |
|---|---|---|---:|---:|---|---:|---:|---|
| YYYY-MM-DD | Qwen2.5 7B | llama.cpp | Q4_K_M | 4096 | técnico | medir | medir | buena |

Comando de medición con llama.cpp:

```bash
/usr/bin/time -l ~/ia-local/llama.cpp/build/bin/llama-cli \
  -m ~/ia-local/models/modelo.gguf \
  -p "Explica en 5 bullets por qué este modelo es adecuado para Mac." \
  -n 200 \
  -ngl 99 \
  -c 4096
```

## Modelos no recomendados localmente en 24 GB

- 30B/32B Q4: normalmente demasiado justo.
- 70B: no viable localmente en 24 GB.
- Full precision 7B+ para inferencia diaria: innecesario.
- Fine-tuning 14B+: mejor cloud.

## Ejercicio práctico

1. Elige un modelo rápido, uno equilibrado y uno ambicioso.
2. Ejecútalos con el mismo prompt.
3. Completa la tabla de medición.
4. Decide tu modelo diario y tu modelo de calidad.

## Recursos

- Ollama library: https://ollama.com/library
- Hugging Face Models: https://huggingface.co/models
- Qwen: https://huggingface.co/Qwen
- Meta Llama: https://www.llama.com/
- Mistral AI: https://huggingface.co/mistralai
- MLX Community: https://huggingface.co/mlx-community

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Chuleta de comandos](C-Chuleta-de-comandos-original-Mac.md) · [Índice](../README.md) · [Troubleshooting: errores comunes y soluciones →](E-Troubleshooting-local.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
