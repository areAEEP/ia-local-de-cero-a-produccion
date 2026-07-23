---
tags:
  - curso/ia-local
  - evaluacion
  - evals
curso: IA-Local-de-Cero-a-Produccion
modulo: anexo-evaluacion
estado: completo
---

# Evaluación de LLMs locales sin autoengaño

<!-- CURSO_NAV_TOP -->
[← P3 - Proyecto - Sistema de serving en producción](../06-Proyectos/04-Sistema-de-serving-en-produccion.md) · [Índice](../README.md) · [Glosario →](B-Glosario.md)
<!-- /CURSO_NAV_TOP -->



> [!goals] Objetivos de aprendizaje
> - Diseñar evaluaciones pequeñas pero útiles.
> - Comparar modelo base, cuantizado, fine-tuned y servido.
> - Evitar decidir por impresiones aisladas.


## Por qué evaluar

Los LLMs son convincentes incluso cuando fallan. Una respuesta bien redactada puede ocultar errores, omisiones o invenciones. Evaluar significa crear fricción contra tu entusiasmo.

No evalúas para obtener una nota académica. Evalúas para decidir:

- qué modelo usar;
- qué quantización aceptar;
- si LoRA compensa;
- si el sistema está listo para usar;
- qué límites debes documentar.

## Baseline primero

Antes de cambiar nada, guarda respuestas del modelo base. Sin baseline, no puedes afirmar mejora.

```text
modelo base
  → respuestas guardadas
modelo modificado
  → mismas preguntas
comparación
```

## Eval mínima manual

Crea 30-50 casos representativos. Para cada caso guarda:

- prompt;
- respuesta ideal o criterios;
- tipo de caso;
- puntuación del modelo base;
- puntuación del modelo candidato;
- observaciones.

Escala simple:

```text
0 = inútil o inventa
1 = parcialmente útil
2 = correcto pero incompleto
3 = correcto, completo y con formato adecuado
```

## Tipos de casos

Incluye variedad:

- casos fáciles;
- casos normales;
- casos ambiguos;
- casos con datos faltantes;
- casos fuera de dominio;
- casos donde debe negarse o pedir aclaración;
- casos largos si tu aplicación tendrá contexto largo.

## Qué medir además de calidad

- Latencia hasta primer token.
- Tokens/segundo.
- Memoria máxima.
- Estabilidad del servidor.
- Cumplimiento de formato.
- Alucinaciones.
- Sensibilidad al prompt.

Un modelo que mejora calidad un 5% pero duplica latencia puede no merecer la pena para agentes.

## Comparaciones justas

Mantén constantes:

- prompt;
- temperatura;
- contexto;
- modelo base cuando sea posible;
- herramienta de ejecución;
- cuantización.

Cambia una variable cada vez.

## Evaluar cuantización

Compara FP16/Q8 si cabe, Q5 y Q4. Busca degradación real, no solo tamaño.

Preguntas:

- ¿pierde formato?
- ¿razona peor?
- ¿repite más?
- ¿alucina más?
- ¿la mejora de velocidad compensa?

Relacionado: [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

## Evaluar LoRA

Separa:

- prompts vistos durante entrenamiento;
- prompts parecidos;
- prompts nuevos;
- prompts fuera de dominio.

Si solo mejora los vistos, memorizó o sobreajustó. Si mejora los nuevos, probablemente aprendió comportamiento.

Relacionado: [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

## Evaluación automática con lm-evaluation-harness

Además de eval manual, puedes usar `lm-evaluation-harness` (EleutherAI) para benchmarks reproducibles.

### Instalación

```bash
cd ~/ia-local/curso
source .venv/bin/activate
uv pip install lm-eval
```

### Benchmark básico con un modelo GGUF vía llama.cpp

Levanta tu servidor local (ver [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md)):

```bash
~/ia-local/llama.cpp/build/bin/llama-server \
  -m ~/ia-local/models/modelo-Q4_K_M.gguf \
  -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
```

En otra terminal, ejecuta un benchmark MMLU reducido:

```bash
lm_eval \
  --model local-chat-completions \
  --model_args "model=local-qwen,base_url=http://127.0.0.1:8080/v1/chat/completions,num_concurrent=1" \
  --tasks mmlu \
  --limit 100 \
  --batch_size 1
```

> [!warning] Recursos
> MMLU completo son miles de preguntas. Usa `--limit 100` o `--limit 500` para pruebas rápidas en local. Un benchmark completo de MMLU puede tardar horas en un 7B local.

### Comparar dos modelos

```bash
# Modelo A
lm_eval --model local-chat-completions \
  --model_args "model=qwen-3b,base_url=http://127.0.0.1:8080/v1/chat/completions" \
  --tasks hellaswag,gsm8k \
  --limit 200 --output_path ~/ia-local/evals/modelo-a

# Modelo B (cambia el servidor al otro modelo y repite)
lm_eval --model local-chat-completions \
  --model_args "model=qwen-7b,base_url=http://127.0.0.1:8080/v1/chat/completions" \
  --tasks hellaswag,gsm8k \
  --limit 200 --output_path ~/ia-local/evals/modelo-b
```

Resultados en JSON:

```bash
cat ~/ia-local/evals/modelo-a/results.json | jq '.results | to_entries[] | {task: .key, acc: .value.acc, acc_norm: .value.acc_norm}' | head -20
```

### Tareas útiles para comparar modelos locales

- `mmlu`: conocimiento general multitema.
- `hellaswag`: comprensión de sentido común.
- `gsm8k`: matemáticas de primaria (razonamiento).
- `arc_challenge`: razonamiento científico.
- `winogrande`: resolución de coreferencia.
- `truthfulqa`: tendencia a alucinar o repetir mitos.

### Con Ollama

Ollama expone `/v1/chat/completions` en versiones recientes:

```bash
lm_eval \
  --model local-chat-completions \
  --model_args "model=qwen2.5:7b-instruct,base_url=http://localhost:11434/v1/chat/completions" \
  --tasks hellaswag \
  --limit 200
```

Verifica primero que Ollama tenga el endpoint OpenAI-compatible activo:

```bash
curl http://localhost:11434/v1/models | jq
```

## Plantilla de eval

```json
{"id":"case-001","tipo":"ambiguo","prompt":"No puedo contabilizar una factura.","criterios":["pide código de error","pide sociedad","no inventa transacción","da pasos numerados"]}
```

Tabla de decisión:

| ID | Base | Candidato | Ganador | Motivo |
|---|---:|---:|---|---|
| case-001 | 1 | 3 | candidato | pide datos faltantes |

## Ejercicio práctico

Crea 20 casos para tu dominio. Evalúa un modelo base y el mismo modelo con otra cuantización. Decide si la cuantización más ligera mantiene calidad suficiente.

## Recursos

- HELM: https://crfm.stanford.edu/helm/
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- OpenAI Evals: https://github.com/openai/evals
- [06-Proyecto-Final](../06-Proyectos/01-Asistente-local-completo.md)

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← P3 - Proyecto - Sistema de serving en producción](../06-Proyectos/04-Sistema-de-serving-en-produccion.md) · [Índice](../README.md) · [Glosario →](B-Glosario.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
