---
tags:
  - curso/ia-local
  - fundamentos
  - llm
  - hardware
curso: IA-Local-de-Cero-a-Produccion
modulo: 01
estado: completo
---

# Fundamentos: qué es un LLM y qué necesita tu equipo

<!-- CURSO_NAV_TOP -->
[← Equivalencias entre Linux, Windows y macOS](../PLATAFORMAS-Y-COMANDOS.md) · [Índice](../README.md) · [Arquitectura transformer - intuición práctica →](02-Arquitectura-transformer.md)
<!-- /CURSO_NAV_TOP -->



> [!TIP]
> **Objetivos de aprendizaje**
> - Entender qué hay dentro de un LLM moderno.
> - Relacionar parámetros, contexto, KV cache y memoria.
> - Diferenciar formatos de pesos: safetensors, GGUF y MLX.
> - Saber qué familias de modelos son realistas según tu RAM, VRAM o memoria unificada.



## Contexto pedagógico: el modelo mental mínimo

Para entender IA local necesitas abandonar la idea de que “el modelo responde”. Más exactamente:

```text
el modelo recibe tokens
  → calcula probabilidades del siguiente token
  → eliges un token con una estrategia de muestreo
  → repites el proceso
```

Todo lo demás —chat, agentes, herramientas, memoria, personalidad— se construye alrededor de ese bucle. Esta distinción es importante porque explica casi todos los límites prácticos:

- La velocidad se mide en tokens/segundo porque se genera token a token.
- La memoria crece con los pesos y con la KV cache.
- El contexto no es comprensión infinita; es una ventana de tokens atendibles.
- La temperatura no cambia el conocimiento del modelo; cambia cómo se muestrea la siguiente palabra.

## Analogía útil

Piensa en un LLM como una función gigantesca:

```text
f(tokens previos) = distribución de probabilidad del siguiente token
```

No guarda recuerdos entre llamadas salvo que tú vuelvas a pasarle el historial dentro del contexto. Por eso los sistemas reales añaden bases de datos, RAG, herramientas o memoria externa. El modelo base solo ve el texto que entra en su ventana.

## Qué debes ser capaz de explicar al terminar

Antes de seguir a [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md), intenta explicar sin comandos:

- por qué un 7B en FP16 ocupa mucho más que un 7B Q4;
- por qué subir contexto consume memoria aunque el modelo sea el mismo;
- por qué una GPU o la memoria unificada ayudan, pero no convierten la memoria disponible en infinita;
- por qué GGUF y MLX son formatos de despliegue, no “modelos diferentes” por sí mismos.

## 1. Qué es un LLM por dentro

Un LLM tipo GPT, Llama, Qwen o Mistral suele ser un transformer decoder-only. A efectos prácticos, contiene:

- **Tokenizer**: convierte texto en tokens.
- **Embeddings**: convierten tokens en vectores.
- **Bloques transformer**: capas repetidas con atención y MLP.
- **Attention heads**: permiten que cada token atienda a tokens anteriores.
- **MLP/feed-forward**: transforma representaciones internas.
- **LM head**: produce probabilidades del siguiente token.

Flujo simplificado:

```text
texto → tokens → embeddings → N bloques transformer → logits → siguiente token
```

Durante inferencia, el modelo no “piensa” todo el texto de golpe como una persona. Predice token a token.

## 2. Parámetros

Los parámetros son los pesos aprendidos del modelo. Un modelo 7B tiene aproximadamente 7.000 millones de números.

En FP16:

```text
7B × 2 bytes ≈ 14 GB solo para pesos
```

Por eso [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md) es clave: reduce el número de bits por peso.

Ejemplos (ver cálculo detallado en [Memoria-KV-Cache-Apple-Silicon](03-Memoria-contexto-y-KV-cache-en-Apple-Silicon.md)):

- 3B FP16: unos 6 GB.
- 7B FP16: unos 14 GB.
- 7B Q4: unos 4-5 GB.
- 14B Q4: unos 8-10 GB.

## 3. Ventana de contexto y KV cache

La ventana de contexto es la cantidad de tokens que el modelo puede tener presentes.

La KV cache guarda claves y valores de atención para no recalcular todo en cada token. A mayor contexto:

- más memoria;
- más latencia;
- posible caída de tokens/segundo.

Un 7B Q4 puede caber bien, pero si pides 32k o 64k tokens de contexto, la KV cache puede ser el problema.

## 4. RAM, VRAM y memoria unificada

### Mac con Apple Silicon

En un Mac M2, CPU y GPU comparten la misma memoria física. Esto ayuda porque no tienes que copiar pesos entre RAM y VRAM separadas, pero también significa que todo compite por la misma bolsa:

- macOS;
- navegador;
- IDE;
- Ollama o llama.cpp;
- KV cache;
- datasets si entrenas.

El ancho de banda de memoria del M2 ronda los 100 GB/s. En inferencia local, muchas veces el cuello real no es “cómputo puro”, sino mover pesos desde memoria.

Consecuencia práctica: los modelos más pequeños y bien cuantizados suelen sentirse mejor que modelos grandes que “caben por los pelos”.

### Windows con GPU dedicada

En un PC con NVIDIA o AMD, la GPU suele tener su propia **VRAM** y el sistema mantiene la **RAM** aparte. El runtime intenta colocar el máximo posible del modelo en la GPU. Si no cabe:

- algunas capas pueden pasar a RAM y CPU;
- el modelo puede cargar, pero generar mucho más despacio;
- el intercambio continuo entre RAM y VRAM puede convertirse en el cuello de botella.

Con 8 GB de VRAM y 32 GB de RAM no tienes una GPU de 40 GB. Tienes dos memorias con velocidades y funciones distintas. Para una primera prueba, elige un modelo cuyos pesos cuantizados quepan en VRAM con margen. Si solo tienes CPU, usa modelos pequeños y acepta una velocidad menor.

### Windows con gráfica integrada

Una gráfica integrada comparte RAM con el sistema, de forma parecida en concepto —no en rendimiento— a la memoria unificada. Vulkan o un runtime que soporte tu iGPU pueden acelerar ciertas cargas, pero no des por hecho que toda la RAM está disponible para el modelo.

## 5. Formatos de pesos

### safetensors

Formato común en Hugging Face. Suele usarse con PyTorch/Transformers.

Ventajas:

- estándar de facto;
- seguro frente a ejecución arbitraria;
- ideal como formato fuente.

Desventajas para inferencia de escritorio:

- puede requerir más memoria;
- no siempre es la ruta más rápida para inferencia.

### GGUF

Formato usado por llama.cpp, Ollama y muchas herramientas de escritorio.

Ventajas:

- excelente para cuantización;
- portable;
- muy práctico en CPU, Metal, CUDA y Vulkan;
- ideal para distribución local.

Lo verás en [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md) y [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

### MLX

Formato/ecosistema de Apple para machine learning en Apple Silicon.

Ventajas:

- muy buen rendimiento en Mac;
- entrenamiento LoRA práctico;
- conversiones directas desde Hugging Face.

Lo usaremos en [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

## 6. Open weights no siempre significa open source

- **Open weights**: puedes descargar pesos del modelo.
- **Open source real**: código, datos o licencia permiten estudiar, modificar y redistribuir ampliamente.

Lee siempre la licencia. Algunas permiten uso comercial; otras no. Algunas imponen políticas de uso o restricciones de redistribución.

## 7. Familias viables según memoria

Empieza por tamaño y después compara familias:

- 8 GB de RAM: modelos de 0,8B–2B con contexto corto;
- 16 GB de RAM o 6–8 GB de VRAM: 3B–4B cuantizado;
- 24–32 GB unificados o 12–16 GB de VRAM: 7B–14B cuantizado;
- 48–64 GB unificados o 24 GB de VRAM: 20B–32B cuantizado, según runtime y contexto.

Qwen, Gemma, Phi, Mistral, Llama y Granite ofrecen puntos de comparación distintos. Consulta la tabla fechada de [Hardware y modelos](../00-Introduccion/01-Elige-hardware-y-modelo.md) en vez de memorizar una lista que pronto quedará antigua.

## 8. Comprobar metadatos de un modelo GGUF

Cuando tengas un GGUF, puedes inspeccionarlo con llama.cpp. Si lo instalaste con Homebrew o winget:

```bash
llama-cli -m /ruta/al/modelo.gguf --prompt "Hola" -n 16
```

En Windows PowerShell:

```powershell
llama-cli.exe -m C:\ruta\al\modelo.gguf --prompt "Hola" -n 16
```

Para listar modelos de Ollama:

```bash
ollama list
ollama show qwen3.5:4b
```

## Profundización: del texto a una distribución de probabilidad

Un LLM no manipula palabras directamente. Manipula tokens. El tokenizer convierte texto en unidades numéricas; esas unidades se convierten en vectores; y los bloques transformer actualizan esos vectores en función del contexto previo.

La salida de la red no es una frase. Es una lista de puntuaciones para todos los tokens posibles. A esas puntuaciones se les suele llamar *logits*. Después, el runtime aplica una estrategia de muestreo para elegir el siguiente token.

```text
texto
  → tokens
  → vectores
  → capas transformer
  → logits
  → softmax/muestreo
  → siguiente token
```

Esto explica por qué el mismo modelo puede responder distinto al mismo prompt si cambias temperatura o semilla. Los pesos no cambian; cambia la selección desde la distribución.

### Atención: la idea que hizo escalar los transformers

La atención permite que cada token combine información de tokens anteriores. En lenguaje natural, muchas palabras dependen de otras que aparecieron antes. En código, una variable puede depender de una definición 100 líneas arriba. En un contrato, una cláusula puede cambiar el significado de otra.

La atención calcula, de forma simplificada:

```text
qué busco  → Query
qué ofrece cada token previo → Key
qué información contiene → Value
```

El modelo compara queries con keys y mezcla values. Por eso hablamos de KV cache: durante generación, guardar keys y values anteriores evita recalcularlo todo.

### Parámetros, capas y capacidad

Más parámetros no significan automáticamente “mejor para mí”. Suelen significar más capacidad para almacenar patrones, pero también:

- más memoria;
- más latencia;
- más coste de cuantización;
- más dificultad para fine-tuning local.

Un 14B puede saber más que un 7B, pero si responde lento, no cabe con contexto o no se integra bien, quizá no sea la mejor herramienta local.

### Contexto no es memoria semántica

La ventana de contexto es texto presente en la llamada actual. No es memoria permanente. Si quieres memoria de proyecto, necesitas pasar información de nuevo o usar un sistema externo.

```text
memoria del modelo: pesos aprendidos durante entrenamiento
contexto: tokens que le pasas ahora
memoria externa: archivos, base vectorial, herramientas, Obsidian, APIs
```

Esta separación evita muchos malentendidos: fine-tuning, RAG y contexto largo resuelven problemas distintos.

### Licencias y procedencia

Los modelos abiertos tienen límites legales y prácticos. Antes de construir sobre uno, revisa:

- permiso de uso comercial;
- requisitos de atribución;
- restricciones de uso;
- si los pesos son derivados de otro modelo;
- si hay tokenizer y configuración completos.

Este punto no es burocrático. Afecta a si puedes desplegar, redistribuir o entrenar derivados.

## Ejercicio práctico

1. Elige tres modelos de [Modelos-Recomendados-24GB](../07-Anexos/D-Modelos-para-Apple-Silicon-24GB.md).
2. Clasifícalos por:
   - tamaño;
   - formato disponible;
   - licencia;
   - si son adecuados para chat, código o razonamiento.
3. Escribe cuál usarías para:
   - asistente general rápido;
   - agente de código;
   - fine-tuning local.

## Recursos

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Hugging Face Models: https://huggingface.co/models
- GGUF en llama.cpp: https://github.com/ggml-org/llama.cpp
- Apple MLX: https://github.com/ml-explore/mlx
- Ollama model library: https://ollama.com/library

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Equivalencias entre Linux, Windows y macOS](../PLATAFORMAS-Y-COMANDOS.md) · [Índice](../README.md) · [Arquitectura transformer - intuición práctica →](02-Arquitectura-transformer.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
