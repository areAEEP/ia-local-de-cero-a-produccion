---
tags:
  - curso/ia-local
  - model-merging
  - interpretabilidad
  - pruning
  - destilacion
curso: IA-Local-de-Cero-a-Produccion
modulo: 05
estado: completo
---

# 05 - Modificación y mejora de pesos

<!-- CURSO_NAV_TOP -->
[← Fine-tuning y adaptación de dominio](02-Fine-tuning-con-PEFT-y-QLoRA.md) · [Índice](../README.md) · [El problema del serving de LLM →](../05-LLMOps/01-El-problema-del-serving.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Linux, Windows y macOS
> Ollama, llama.cpp y Python sirven en los tres sistemas. MLX y Metal son exclusivos de Apple Silicon; en Linux usa CUDA, ROCm/HIP, Vulkan o CPU según tu equipo. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!goals] Objetivos de aprendizaje
> - Conocer técnicas más allá del fine-tuning: merging, ablación, pruning, destilación y tokenizers.
> - Entender qué es viable en un Mac M2 con 24 GB y qué conviene hacer en cloud.
> - Practicar un merge acotado con modelos pequeños o 7B.
> - Saber cuándo estas técnicas aportan valor y cuándo son complejidad innecesaria.



## Contexto pedagógico: cirugía de modelos con humildad

Modificar pesos más allá de LoRA es tentador porque parece que puedes “diseñar” capacidades. En la práctica, estás manipulando sistemas de alta dimensión donde los efectos colaterales son normales.

La idea central de este módulo es distinguir entre:

```text
cambios controlados y evaluables
vs
cambios llamativos pero difíciles de justificar
```

Un merge puede mejorar un benchmark y empeorar tu caso real. Una ablación puede reducir un comportamiento y dañar otro. Un pruning puede ahorrar memoria y perder calidad. Por eso toda cirugía de pesos debe ir unida a evaluación.

## Intuición de model merging

Si dos modelos comparten arquitectura, sus pesos viven en espacios comparables. Un merge intenta combinar “direcciones” aprendidas por cada uno.

Analogía limitada pero útil:

```text
modelo A: buen seguimiento de instrucciones
modelo B: mejor en código
merge: intento de conservar parte de ambos
```

No es una suma garantizada de habilidades. Los modelos pueden interferir. Por eso métodos como TIES o DARE intentan reducir conflictos entre deltas.

## Intuición de destilación

La destilación no copia pesos. Copia comportamiento observado. Generas ejemplos con un profesor y entrenas un alumno para imitar respuestas.

```text
profesor grande → respuestas de entrenamiento → alumno pequeño
```

Esto tiene sentido cuando quieres velocidad local y puedes aceptar que el alumno solo aprenda una parte del comportamiento del profesor.

## Criterio práctico

Si tu objetivo es que algo quepa en 24 GB, empieza por [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md). Si tu objetivo es adaptar formato o procedimiento, empieza por [04-Fine-Tuning](01-Fine-tuning-con-MLX-en-Mac.md). Usa merging/cirugía solo cuando puedas explicar qué mejora esperas y cómo la medirás.

## 1. Mapa de técnicas

Después de [04-Fine-Tuning](01-Fine-tuning-con-MLX-en-Mac.md), puedes modificar o mejorar modelos de varias formas:

- **Model merging**: combinar pesos de modelos compatibles.
- **Ablación de activaciones**: localizar y reducir direcciones de comportamiento.
- **Pruning**: eliminar pesos, capas o componentes.
- **Destilación**: entrenar un modelo pequeño con salidas de uno mayor.
- **Extensión de tokenizer**: añadir tokens o vocabulario especial.

En 24 GB, las prácticas realistas son merges pequeños, pruebas conceptuales y evaluación. Para cirugía pesada, usa cloud.

## 2. Model merging con mergekit

Mergear combina modelos de la misma arquitectura o compatibles. Métodos habituales:

- **Linear merge**: promedio ponderado simple.
- **SLERP**: interpolación esférica; útil para mezclar dos modelos.
- **TIES**: intenta resolver interferencias entre deltas.
- **DARE**: aplica dropout/rescale sobre deltas para preservar capacidades.

Instala:

```bash
cd ~/ia-local/curso
source .venv/bin/activate
uv pip install mergekit
```

Ejemplo de configuración SLERP:

```bash
mkdir -p ~/ia-local/merges
cat > ~/ia-local/merges/slerp.yml <<'EOF'
models:
  - model: Qwen/Qwen2.5-3B-Instruct
  - model: Qwen/Qwen2.5-3B
merge_method: slerp
base_model: Qwen/Qwen2.5-3B
parameters:
  t:
    - value: 0.5
dtype: float16
EOF
```

Ejecuta:

```bash
mergekit-yaml ~/ia-local/merges/slerp.yml ~/ia-local/merges/qwen3b-slerp --copy-tokenizer
```

Este ejemplo puede descargar varios GB. En 24 GB es más razonable hacerlo con 3B. Con 7B puede funcionar, pero será lento y consumirá bastante disco/RAM. Para 14B, usa cloud.

## 3. Convertir el merge a GGUF

```bash
cd ~/ia-local/llama.cpp
source ~/ia-local/curso/.venv/bin/activate
python convert_hf_to_gguf.py ~/ia-local/merges/qwen3b-slerp \
  --outfile ~/ia-local/merges/qwen3b-slerp-f16.gguf \
  --outtype f16

./build/bin/llama-quantize \
  ~/ia-local/merges/qwen3b-slerp-f16.gguf \
  ~/ia-local/merges/qwen3b-slerp-Q4_K_M.gguf \
  Q4_K_M
```

Prueba:

```bash
./build/bin/llama-cli \
  -m ~/ia-local/merges/qwen3b-slerp-Q4_K_M.gguf \
  -p "Di qué capacidades esperas de este modelo fusionado." \
  -n 150 \
  -ngl 99
```

## 4. Interpretabilidad práctica y ablación

La interpretabilidad busca entender qué direcciones, neuronas o capas se asocian a comportamientos.

La “abliteration” o ablación de direcciones consiste en:

1. recopilar prompts positivos/negativos;
2. medir activaciones;
3. identificar una dirección asociada al comportamiento;
4. proyectar o reducir esa dirección;
5. evaluar regresiones.

Esto puede servir para investigación, pero tiene riesgos:

- degradar capacidades generales;
- eliminar señales útiles;
- producir efectos colaterales difíciles de detectar;
- cruzar límites de seguridad o licencia.

En este curso no hacemos ablación agresiva local. La tratamos como técnica conceptual. Si la pruebas, usa modelos pequeños, datasets controlados y evals claras.

## 5. Pruning

Pruning elimina partes del modelo:

- pesos individuales;
- canales;
- attention heads;
- capas completas.

Puede reducir coste, pero no es gratis. Después de podar suele hacer falta reentrenar o destilar.

En 24 GB, pruning serio de LLMs no es el mejor uso del tiempo. Si tu objetivo es reducir memoria, [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md) suele dar mejor retorno.

## 6. Destilación

Destilar es entrenar un modelo pequeño para imitar salidas de un modelo grande.

Pipeline típico:

```text
prompts propios → respuestas de modelo profesor → dataset → fine-tuning de modelo alumno → eval
```

Ejemplo de generación de dataset con Ollama:

```bash
mkdir -p ~/ia-local/distill
cat > ~/ia-local/distill/prompts.txt <<'EOF'
Explica qué es GGUF.
Compara Ollama y llama.cpp.
Da una regla para elegir cuantización.
EOF

while read -r prompt; do
  jq -n --arg p "$prompt" '{model:"qwen2.5:7b-instruct", prompt:$p, stream:false}' \
  | curl -s http://localhost:11434/api/generate -H 'Content-Type: application/json' -d @- \
  | jq -c --arg p "$prompt" '{messages:[{role:"system","content":"Eres un asistente técnico directo."},{role:"user",content:$p},{role:"assistant",content:.response}]}'
done < ~/ia-local/distill/prompts.txt > ~/ia-local/distill/train.jsonl
```

Este dataset es demasiado pequeño; sirve para ver el formato. Para algo útil, genera y revisa cientos o miles de ejemplos.

## 7. Extensión de tokenizer y vocabulario

Añadir tokens especiales puede tener sentido si necesitas:

- etiquetas estructurales;
- tokens de dominio repetitivos;
- formatos internos de herramientas.

Pero no es inocuo. Al añadir tokens, normalmente tienes que redimensionar embeddings y entrenar. Para la mayoría de casos locales, basta con buen prompting o LoRA.

## 8. Evaluar merges y cirugía

Cualquier modificación de pesos exige eval antes/después:

```text
base → modelo modificado → modelo cuantizado → despliegue
```

Eval mínima:

- 20 prompts de conocimiento general;
- 20 prompts del dominio;
- 10 prompts de formato;
- 10 prompts adversariales o de límites;
- medición de velocidad y memoria.

Si no puedes demostrar mejora, no asumas que el modelo es mejor.

## Profundización: por qué los merges pueden fallar

Dos modelos pueden tener la misma arquitectura y aun así no combinar bien. Durante entrenamiento, cada modelo llega a una solución interna distinta. Aunque ambos funcionen, sus pesos pueden codificar habilidades de forma no perfectamente alineada.

Un merge ingenuo puede producir:

- pérdida de formato instruct;
- respuestas incoherentes;
- degradación en razonamiento;
- mezcla de estilos incompatible;
- aparente mejora en unos prompts y regresión en otros.

Por eso los métodos de merge no son magia. Son hipótesis sobre cómo combinar deltas de pesos minimizando interferencias.

### SLERP, TIES y DARE con intuición

- **SLERP** interpola entre dos puntos de forma suave. Útil cuando quieres un punto intermedio entre modelos similares.
- **TIES** intenta conservar cambios importantes y resolver signos conflictivos entre deltas.
- **DARE** introduce descarte/reescala para reducir interferencias y preservar capacidades.

La decisión no debería ser estética. Debería venir de una evaluación: ¿qué método conserva mejor lo que te importa?

### Interpretabilidad: cuidado con las narrativas bonitas

Es fácil mirar una activación, una dirección o una capa y contar una historia convincente. Pero en modelos grandes, las representaciones son distribuidas. Una dirección puede correlacionar con un comportamiento sin ser “la causa única”.

Por eso cualquier intervención debe validarse con:

- prompts positivos;
- prompts negativos;
- pruebas de regresión;
- tareas generales;
- medición de daño colateral.

### Pruning y destilación frente a cuantización

Si el objetivo es ejecutar en 24 GB, la cuantización suele ser la primera herramienta. Pruning y destilación son más complejos porque pueden requerir reentrenamiento o generación de datos.

Orden práctico recomendado:

```text
1. elegir modelo menor
2. cuantizar mejor
3. optimizar contexto/runtime
4. LoRA si necesitas comportamiento
5. destilación/merge/cirugía solo si hay hipótesis clara
```

### Ética y licencias

Modificar pesos puede crear derivados sujetos a licencias del modelo base. Antes de compartir un merge o adaptador, revisa si la licencia lo permite. Además, técnicas de ablación o eliminación de restricciones pueden tener implicaciones de seguridad; trátalas como investigación controlada, no como atajo para producción.

## Ejercicio práctico

1. Instala mergekit.
2. Haz un merge pequeño con modelos 3B compatibles.
3. Convierte el resultado a GGUF Q4_K_M.
4. Compara 10 prompts contra el modelo base.
5. Decide si el merge aporta algo medible.

Si tu Mac se queda corto, no fuerces: haz el merge en cloud o baja a modelos 1.5B-3B.

## Recursos

- mergekit: https://github.com/arcee-ai/mergekit
- TIES-Merging: https://arxiv.org/abs/2306.01708
- DARE: https://arxiv.org/abs/2311.03099
- Distilling the Knowledge in a Neural Network: https://arxiv.org/abs/1503.02531
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Fine-tuning y adaptación de dominio](02-Fine-tuning-con-PEFT-y-QLoRA.md) · [Índice](../README.md) · [El problema del serving de LLM →](../05-LLMOps/01-El-problema-del-serving.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
