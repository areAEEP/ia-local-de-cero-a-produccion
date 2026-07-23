---
tags:
  - curso/ia-local
  - glosario
curso: IA-Local-de-Cero-a-Produccion
modulo: anexo-glosario
estado: completo
---

# Glosario

<!-- CURSO_NAV_TOP -->
[← Evaluación de LLMs locales sin autoengaño](A-Evaluacion-local-sin-autoengano.md) · [Índice](../README.md) · [Chuleta de comandos →](C-Chuleta-de-comandos-original-Mac.md)
<!-- /CURSO_NAV_TOP -->



> [!TIP]
> **Objetivos de aprendizaje**
> - Tener definiciones breves de los conceptos usados en el curso.
> - Saltar rápido al módulo donde se explica cada tema.


## Términos

**Ablación**  
Técnica para reducir o eliminar una señal interna del modelo. Ver [05-Modificacion-Pesos](../04-Adaptar/03-Merging-pruning-y-destilacion.md).

**Agent**  
Sistema que combina un LLM con herramientas externas para realizar tareas que el modelo no puede por sí solo. Ver [08-Agentes-Locales](../03-Construir/02-Agentes-locales-y-MCP.md).

**Attention (atención)**  
Mecanismo que permite a cada token combinar información de tokens anteriores mediante Query, Key y Value. Ver [Arquitectura-Transformer-Intuicion](../01-Fundamentos/02-Arquitectura-transformer.md).

**AWQ**  
Método de cuantización orientado a inferencia eficiente, frecuente en GPU NVIDIA. En Mac priorizamos GGUF o MLX. Ver [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

**Baseline**  
Resultado de referencia antes de cambiar modelo, prompt, cuantización o entrenamiento. Ver [Evaluacion-LLMs-Local](A-Evaluacion-local-sin-autoengano.md).

**BF16 / FP16**  
Formatos de coma flotante de 16 bits. Usados en entrenamiento e inferencia sin cuantización agresiva. Ver [01-Fundamentos](../01-Fundamentos/01-Que-es-un-LLM.md).

**Chunking**  
División de documentos en fragmentos para recuperación semántica en RAG. Ver [07-RAG-Local](../03-Construir/01-RAG-local.md).

**Contexto**  
Número de tokens que el modelo puede considerar. Más contexto consume más KV cache. Ver [01-Fundamentos](../01-Fundamentos/01-Que-es-un-LLM.md) y [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md).

**Decode**  
Fase token a token de generación después del prefill. Ver [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md).

**Destilación**  
Entrenar un modelo pequeño para imitar a uno mayor. Ver [05-Modificacion-Pesos](../04-Adaptar/03-Merging-pruning-y-destilacion.md).

**Embeddings**  
Vectores que representan tokens o documentos en un espacio matemático de alta dimensión. Se usan tanto en el transformer como en RAG para recuperación semántica. Ver [Arquitectura-Transformer-Intuicion](../01-Fundamentos/02-Arquitectura-transformer.md) y [07-RAG-Local](../03-Construir/01-RAG-local.md).

**Flash attention**  
Optimización del cálculo de atención que reorganiza accesos a memoria sin cambiar los resultados. Se activa con `-fa` en llama.cpp. Ver [Arquitectura-Transformer-Intuicion](../01-Fundamentos/02-Arquitectura-transformer.md).

**GGUF**  
Formato de pesos usado por llama.cpp, Ollama y LM Studio. Muy práctico para cuantización local. Ver [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

**GPTQ**  
Método de cuantización post-entrenamiento habitual en GPU NVIDIA. Ver [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

**GQA (Grouped-Query Attention)**  
Optimización donde grupos de heads comparten K y V, reduciendo el KV cache. Ver [Arquitectura-Transformer-Intuicion](../01-Fundamentos/02-Arquitectura-transformer.md) y [Memoria-KV-Cache-Apple-Silicon](../01-Fundamentos/03-Memoria-contexto-y-KV-cache-en-Apple-Silicon.md).

**KV cache**  
Memoria que guarda keys y values de atención para acelerar generación. Crece con el contexto. Ver [Memoria-KV-Cache-Apple-Silicon](../01-Fundamentos/03-Memoria-contexto-y-KV-cache-en-Apple-Silicon.md).

**Layer normalization**  
Normalización de activaciones dentro de cada bloque transformer para estabilizar el entrenamiento y la inferencia. Ver [Arquitectura-Transformer-Intuicion](../01-Fundamentos/02-Arquitectura-transformer.md).

**Logits**  
Puntuaciones sin normalizar que el modelo produce para cada token posible antes del muestreo. Ver [Arquitectura-Transformer-Intuicion](../01-Fundamentos/02-Arquitectura-transformer.md).

**LoRA**  
Low-Rank Adaptation. Entrena adaptadores pequeños en vez de todos los pesos. Ver [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

**MCP (Model Context Protocol)**  
Protocolo estándar para que LLMs se conecten a herramientas y fuentes de datos externas. Ver [08-Agentes-Locales](../03-Construir/02-Agentes-locales-y-MCP.md).

**Metal / MPS**  
API de gráficos y cómputo de Apple. Es el backend que usan llama.cpp y MLX para aprovechar la GPU de Apple Silicon. Ver 00-Setup.

**MLP (Feed-Forward)**  
Capa de transformación no lineal dentro de cada bloque transformer. Gran parte del conocimiento se expresa aquí. Ver [Arquitectura-Transformer-Intuicion](../01-Fundamentos/02-Arquitectura-transformer.md).

**MLX**  
Framework de Apple para machine learning en Apple Silicon. Ver 00-Setup, [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md) y [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

**Model merging**  
Combinar pesos de modelos compatibles para intentar conservar capacidades de ambos. Ver [05-Modificacion-Pesos](../04-Adaptar/03-Merging-pruning-y-destilacion.md).

**Modelfile**  
Archivo de Ollama para crear modelos personalizados con parámetros, sistema y pesos. Ver [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md) y [06-Proyecto-Final](../06-Proyectos/01-Asistente-local-completo.md).

**Multi-head attention**  
Atención con múltiples heads paralelos, cada uno enfocando distintos aspectos de la secuencia. Ver [Arquitectura-Transformer-Intuicion](../01-Fundamentos/02-Arquitectura-transformer.md).

**Open weights**  
Pesos descargables, aunque no necesariamente open source completo. Ver [01-Fundamentos](../01-Fundamentos/01-Que-es-un-LLM.md).

**Perplexity**  
Métrica de predicción de lenguaje. Menor suele ser mejor, pero no sustituye una evaluación de tarea real. Ver [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

**Positional encoding**  
Información de posición añadida a los embeddings para que el modelo distinga el orden de los tokens. Ver [Arquitectura-Transformer-Intuicion](../01-Fundamentos/02-Arquitectura-transformer.md).

**Prefill**  
Fase inicial de inferencia donde el modelo procesa el prompt completo antes de generar tokens nuevos. Ver [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md).

**Pruning**  
Eliminación de pesos, heads o capas para reducir tamaño o coste. Suele requerir reentrenamiento. Ver [05-Modificacion-Pesos](../04-Adaptar/03-Merging-pruning-y-destilacion.md).

**QLoRA**  
LoRA sobre modelo base cuantizado. Reduce memoria de entrenamiento. Ver [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

**Quantization scheme**  
Esquema concreto de cuantización, por ejemplo Q4_K_M o Q5_K_M. Ver [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

**RAG (Retrieval-Augmented Generation)**  
Técnica que combina recuperación de documentos relevantes con generación del LLM. Ver [07-RAG-Local](../03-Construir/01-RAG-local.md).

**Rank**  
Dimensión interna del adaptador LoRA. Más rank implica más capacidad y más coste. Ver [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

**ReAct**  
Patrón de agente que alterna razonamiento y acciones (Reason-Act-Observe) para resolver tareas con herramientas. Ver [08-Agentes-Locales](../03-Construir/02-Agentes-locales-y-MCP.md).

**Reranking**  
Reordenación de resultados de recuperación con un modelo más preciso para mejorar la calidad del RAG. Ver [07-RAG-Local](../03-Construir/01-RAG-local.md).

**Safetensors**  
Formato de pesos seguro y eficiente usado en Hugging Face con PyTorch/Transformers. Ver [01-Fundamentos](../01-Fundamentos/01-Que-es-un-LLM.md).

**SLERP**  
Interpolación esférica para mergear modelos. Ver [05-Modificacion-Pesos](../04-Adaptar/03-Merging-pruning-y-destilacion.md).

**Sobreajuste (overfitting)**  
Cuando un fine-tune aprende demasiado los ejemplos concretos y generaliza mal a casos nuevos. Ver [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

**Tokenizer**  
Componente que convierte texto en tokens. Ver [01-Fundamentos](../01-Fundamentos/01-Que-es-un-LLM.md).

**Tool call**  
Invocación de una herramienta externa desde un agente LLM. Ver [08-Agentes-Locales](../03-Construir/02-Agentes-locales-y-MCP.md).

**VLM (Vision-Language Model)**  
Modelo que procesa tanto texto como imágenes. Ver [Multimodal-Local](../03-Construir/03-IA-multimodal-local.md).

**Whisper**  
Modelo de reconocimiento de voz de OpenAI, optimizable localmente con whisper.cpp o mlx-whisper. Ver [Whisper-STT-Local](../03-Construir/04-Voz-y-transcripcion-local.md).


## Ejercicio práctico

Elige 10 términos y escribe una explicación de una frase con tus palabras. Después vuelve al módulo correspondiente y corrige lo que hayas simplificado demasiado.

## Recursos

- Hugging Face glossary: https://huggingface.co/docs/transformers/glossary
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- llama.cpp: https://github.com/ggml-org/llama.cpp

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Evaluación de LLMs locales sin autoengaño](A-Evaluacion-local-sin-autoengano.md) · [Índice](../README.md) · [Chuleta de comandos →](C-Chuleta-de-comandos-original-Mac.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
