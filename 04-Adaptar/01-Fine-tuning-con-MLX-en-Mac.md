---
tags:
  - curso/ia-local
  - fine-tuning
  - lora
  - qlora
  - mlx
curso: IA-Local-de-Cero-a-Produccion
modulo: 04
estado: completo
---

# 04 - Fine-tuning en Mac

<!-- CURSO_NAV_TOP -->
[← Voz y transcripción local con Whisper](../03-Construir/04-Voz-y-transcripcion-local.md) · [Índice](../README.md) · [Fine-tuning y adaptación de dominio →](02-Fine-tuning-con-PEFT-y-QLoRA.md)
<!-- /CURSO_NAV_TOP -->



> [!NOTE]
> **Ruta Apple Silicon**
> Este capítulo nació para Mac con Apple Silicon. Si usas Windows, quédate con los conceptos y sigue la alternativa indicada en [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!TIP]
> **Objetivos de aprendizaje**
> - Entender por qué full fine-tuning no es realista en un Mac M2 24 GB.
> - Aplicar LoRA/QLoRA como vía práctica.
> - Preparar datasets JSONL en formato chat/instruct.
> - Entrenar con MLX, fusionar adaptadores y exportar a GGUF/Ollama.
> - Evaluar honestamente si el ajuste mejora el modelo base.



## Contexto pedagógico: entrenar no es “meter conocimiento” sin más

Fine-tuning significa modificar el comportamiento del modelo ajustando pesos o adaptadores. Pero no conviene verlo como una base de datos comprimida. Es más útil pensarlo como una forma de cambiar probabilidades:

```text
dados prompts parecidos a estos ejemplos
  → aumenta la probabilidad de respuestas con este estilo/formato/criterio
```

Si quieres que el modelo recuerde información cambiante, usa RAG o herramientas. Si quieres que responda con una estructura, tono o procedimiento estable, LoRA puede tener sentido.

## Por qué LoRA funciona

Una red grande tiene muchas matrices. LoRA no reentrena esas matrices completas. Añade una actualización de bajo rango:

```text
W_original + ΔW
ΔW ≈ A × B, donde A y B son matrices pequeñas
```

Eso reduce muchísimo el número de parámetros entrenables. La idea pedagógica clave es que no estás creando un modelo desde cero; estás empujando un modelo existente en una dirección concreta.

## Qué aprende un buen LoRA

Un buen LoRA puede aprender:

- formato de respuesta;
- vocabulario de dominio;
- prioridades de un procedimiento;
- estilo de interacción;
- patrones repetidos de razonamiento corto.

Un LoRA pequeño no suele aprender bien:

- conocimiento masivo nuevo;
- hechos raros con pocos ejemplos;
- razonamiento muy superior al modelo base;
- datos cambiantes o privados que deberían estar en una herramienta.

## La pregunta antes de entrenar

Antes de ejecutar un comando, formula la hipótesis:

```text
Creo que el fine-tune mejorará X, y lo mediré con Y.
```

Ejemplo bueno:

```text
Mejorará el formato de respuestas de soporte: más pasos numerados, menos divagación y más petición de datos faltantes. Lo mediré con 40 casos no vistos.
```

Si no puedes escribir esa frase, todavía no estás listo para entrenar.

## 1. Full fine-tuning vs LoRA

Full fine-tuning actualiza todos los pesos del modelo. En un 7B eso implica pesos, gradientes, optimizador y activaciones. En 24 GB no es una opción cómoda.

LoRA entrena matrices pequeñas de bajo rango que se acoplan a capas del modelo. En vez de cambiar todo el modelo, aprendes un adaptador.

Ventajas:

- mucha menos memoria;
- entrenamiento viable en Apple Silicon;
- adaptadores pequeños;
- puedes fusionar después si quieres desplegar.

QLoRA añade cuantización del modelo base durante el entrenamiento. En Mac, la ruta práctica suele ser MLX con modelos ya preparados en 4-bit.

## 2. Cuándo tiene sentido fine-tunear

Fine-tuning no es magia. Úsalo para:

- formato de respuesta consistente;
- estilo o tono específico;
- conocimiento de procedimientos relativamente estable;
- clasificación o extracción con ejemplos repetibles.

No lo uses para:

- meter datos que cambian cada día;
- sustituir una base de conocimiento;
- hacer que un 3B razone como un 70B;
- corregir prompts malos.

Para conocimiento cambiante, usa RAG o herramientas. Para comportamiento, LoRA puede funcionar bien.

## 3. Preparar dataset JSONL

Crea carpetas:

```bash
mkdir -p ~/ia-local/fine-tuning/data
cd ~/ia-local/fine-tuning
```

Formato recomendado para MLX chat:

```json
{"messages":[{"role":"system","content":"Eres un asistente técnico directo."},{"role":"user","content":"¿Qué es GGUF?"},{"role":"assistant","content":"GGUF es un formato de pesos usado por llama.cpp y herramientas locales para cargar modelos cuantizados de forma eficiente."}]}
```

Crea `train.jsonl`:

```bash
cat > ~/ia-local/fine-tuning/data/train.jsonl <<'EOF'
{"messages":[{"role":"system","content":"Eres un asistente técnico directo."},{"role":"user","content":"¿Cuándo uso Ollama?"},{"role":"assistant","content":"Usa Ollama cuando quieras ejecutar y servir modelos locales con poca configuración."}]}
{"messages":[{"role":"system","content":"Eres un asistente técnico directo."},{"role":"user","content":"¿Cuándo uso llama.cpp?"},{"role":"assistant","content":"Usa llama.cpp cuando necesites control fino sobre GGUF, Metal, contexto, batch y servidor local."}]}
EOF
```

Para un entrenamiento real necesitas más ejemplos. Mínimo práctico: 200-500 ejemplos buenos. Mejor 1.000 ejemplos limpios que 20.000 mediocres.

Divide validación:

```bash
cp ~/ia-local/fine-tuning/data/train.jsonl ~/ia-local/fine-tuning/data/valid.jsonl
```

Esto solo sirve para smoke test. En serio, separa ejemplos distintos.

## 4. Entrenar LoRA con MLX

Si no los tienes ya del 00-Setup, instala las dependencias:

```bash
cd ~/ia-local/curso
source .venv/bin/activate
uv pip install mlx mlx-lm huggingface_hub datasets transformers sentencepiece protobuf
```

Smoke test con 3B:

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --train \
  --data ~/ia-local/fine-tuning/data \
  --iters 100 \
  --batch-size 1 \
  --lora-layers 16 \
  --learning-rate 1e-5 \
  --adapter-path ~/ia-local/fine-tuning/adapters/qwen3b-lora
```

Para 7B en 24 GB:

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --train \
  --data ~/ia-local/fine-tuning/data \
  --iters 300 \
  --batch-size 1 \
  --lora-layers 16 \
  --learning-rate 1e-5 \
  --adapter-path ~/ia-local/fine-tuning/adapters/qwen7b-lora
```

Si hay presión de memoria:

- baja a 3B;
- reduce `--lora-layers`;
- reduce longitud de ejemplos;
- cierra apps;
- usa cloud para modelos mayores.

## 5. Hiperparámetros prácticos

- `rank`: capacidad del adaptador. 8-16 para empezar.
- `alpha`: escala LoRA. Suele ir ligado al rank.
- `learning_rate`: empieza en `1e-5` o `2e-5`.
- `iters`: 100 para smoke test; 500-2000 para algo serio.
- `batch-size`: en 24 GB, empieza con 1.

Señales de problema:

- loss baja demasiado rápido y eval empeora: sobreajuste;
- respuestas clónicas: dataset repetitivo;
- olvida capacidades generales: entrenamiento agresivo o dataset estrecho.

## 6. Probar el adaptador

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --adapter-path ~/ia-local/fine-tuning/adapters/qwen3b-lora \
  --prompt "¿Cuándo uso llama.cpp?" \
  --max-tokens 120 \
  --temp 0.2
```

Compara contra el modelo base sin adaptador:

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --prompt "¿Cuándo uso llama.cpp?" \
  --max-tokens 120 \
  --temp 0.2
```

## 7. Fusionar adaptador

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --adapter-path ~/ia-local/fine-tuning/adapters/qwen3b-lora \
  --save-path ~/ia-local/fine-tuning/fused/qwen3b-curso
```

Después puedes servirlo con MLX:

```bash
mlx_lm.server \
  --model ~/ia-local/fine-tuning/fused/qwen3b-curso \
  --host 127.0.0.1 \
  --port 8081
```

## 8. Exportar a GGUF para Ollama

El flujo suele ser:

```text
MLX fused → convertir a Hugging Face si procede → convert_hf_to_gguf.py → llama-quantize → Ollama Modelfile
```

Dependiendo de versión, MLX puede cambiar comandos de exportación. Verifica ayuda:

```bash
mlx_lm.fuse --help
mlx_lm.convert --help
```

Si necesitas despliegue simple, puedes saltarte GGUF y servir con `mlx_lm.server`. Si necesitas Ollama, usa llama.cpp para convertir/cuanti­zar como en [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

## 9. Cuándo saltar a cloud

Usa RunPod, Vast.ai, Lambda, Colab o similar si:

- quieres entrenar 13B/14B con margen;
- necesitas batches mayores;
- tu dataset es grande;
- quieres iterar rápido;
- necesitas full fine-tuning.

Estrategia sensata: entrenar adaptador en cloud y traer solo el adaptador/modelo final al Mac.

## 10. Evaluación honesta

No te fíes de “parece mejor”. Crea un eval pequeño:

```json
{"prompt":"¿Cuándo uso Ollama?","ideal":"Cuando quiera ejecutar y servir modelos locales con poca configuración."}
```

Guarda 20-50 prompts que representen tu caso real. Compara:

- modelo base;
- modelo con LoRA;
- modelo fusionado/cuantizado.

Mide:

- exactitud;
- formato;
- alucinaciones;
- latencia;
- regresiones.

## Profundización: datos, pérdida y generalización

Durante fine-tuning, el modelo intenta reducir una función de pérdida: básicamente, que la respuesta esperada sea más probable dado el prompt. Si tus ejemplos son buenos, empujas el comportamiento en una dirección útil. Si son malos, enseñas ruido.

La calidad del dataset pesa más que el número bruto de ejemplos. Un dataset pequeño, consistente y revisado puede superar a uno enorme con respuestas contradictorias.

### Qué significa sobreajuste en un LoRA

Sobreajuste no es solo memorizar frases. En un LoRA puede verse como:

- repetir estructuras demasiado rígidas;
- responder con el estilo del dataset aunque no encaje;
- perder capacidad general;
- copiar detalles de ejemplos vistos;
- fallar en prompts ligeramente distintos.

Se detecta separando entrenamiento, validación y evaluación. Si pruebas solo con prompts parecidos a los de entrenamiento, te engañas.

### Formato chat: por qué importa

Los modelos instruct han sido entrenados para seguir plantillas de conversación. Si tu JSONL no respeta roles (`system`, `user`, `assistant`) o mezcla estilos, el modelo recibe señales confusas.

Un buen ejemplo de entrenamiento debe tener:

- instrucción clara;
- entrada representativa;
- respuesta ideal;
- formato consistente;
- ausencia de información que no quieras que reproduzca.

### LoRA como cambio de comportamiento, no como almacén

Si quieres que un asistente conozca 500 procedimientos que cambian cada mes, mejor usa documentos y recuperación. Si quieres que, al consultar procedimientos, responda siempre con pasos, riesgos y datos faltantes, LoRA es adecuado.

```text
conocimiento cambiante → RAG/herramientas
estilo y protocolo estable → LoRA
capacidad base insuficiente → modelo mejor
```

### Evaluación antes de gastar horas

Antes de entrenar, guarda respuestas del modelo base. Después del LoRA, compara contra esas respuestas. Si no tienes baseline, no sabrás si mejoraste.

Métrica manual simple:

```text
0 = falla o inventa
1 = parcialmente útil
2 = correcto pero incompleto
3 = correcto, completo y con formato deseado
```

Evalúa prompts vistos y no vistos. Los no vistos son los que importan.

## Ejercicio práctico

1. Crea 20 ejemplos JSONL sobre un dominio sencillo.
2. Entrena LoRA 3B durante 100 iteraciones.
3. Prueba 5 prompts vistos y 5 no vistos.
4. Escribe si mejoró realmente y qué empeoró.
5. Si el resultado es prometedor, repite con más datos y 300-500 iteraciones.

## Recursos

- MLX LM LoRA: https://github.com/ml-explore/mlx-lm
- LoRA paper: https://arxiv.org/abs/2106.09685
- QLoRA paper: https://arxiv.org/abs/2305.14314
- Hugging Face datasets: https://huggingface.co/docs/datasets
- RunPod: https://www.runpod.io/
- Vast.ai: https://vast.ai/

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Voz y transcripción local con Whisper](../03-Construir/04-Voz-y-transcripcion-local.md) · [Índice](../README.md) · [Fine-tuning y adaptación de dominio →](02-Fine-tuning-con-PEFT-y-QLoRA.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
