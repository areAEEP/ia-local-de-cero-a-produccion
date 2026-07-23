---
tags:
  - curso/ia-local
  - inferencia
  - ollama
  - llama-cpp
  - mlx
curso: IA-Local-de-Cero-a-Produccion
modulo: 02
estado: completo
---

# 02 - Inferencia local

<!-- CURSO_NAV_TOP -->
[← Memoria, contexto y KV cache](../01-Fundamentos/03-Memoria-contexto-y-KV-cache-en-Apple-Silicon.md) · [Índice](../README.md) · [03 - Cuantización y reducción de memoria →](02-Cuantizacion-y-formatos.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Linux, Windows y macOS
> Ollama, llama.cpp y Python sirven en los tres sistemas. MLX y Metal son exclusivos de Apple Silicon; en Linux usa CUDA, ROCm/HIP, Vulkan o CPU según tu equipo. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!goals] Objetivos de aprendizaje
> - Ejecutar modelos locales con Ollama, llama.cpp y MLX.
> - Entender parámetros de generación: temperature, top_p, repeat penalty y contexto.
> - Levantar una API local compatible con OpenAI.
> - Integrar un modelo local con aplicaciones, agentes y servidores MCP.



## Contexto pedagógico: qué significa inferir

Inferencia es usar un modelo ya entrenado para generar salida. No estás cambiando sus pesos. Estás haciendo pasar tokens por una red neuronal y aplicando una estrategia de generación.

El ciclo completo es:

```text
prompt
  → tokenización
  → carga/uso de pesos
  → cálculo de logits
  → muestreo del siguiente token
  → actualización de KV cache
  → repetición hasta terminar
```

Cuando cambias de Ollama a llama.cpp o MLX, no cambia la idea central. Cambia el runtime: cómo se cargan los pesos, cómo se usa Metal, qué formato espera y cuánto control te da.

## Qué estás observando en este módulo

Cada prueba de inferencia debe responder a cuatro preguntas:

1. **¿Cabe?** Memoria usada por pesos + KV cache + sistema.
2. **¿Va suficientemente rápido?** Tokens/segundo y latencia inicial.
3. **¿Responde con calidad suficiente?** No solo fluidez; también precisión y formato.
4. **¿Es integrable?** CLI, API local, agente o aplicación.

La inferencia local es una negociación constante entre esas cuatro variables. Un modelo grande que responde muy bien pero tarda demasiado puede ser peor herramienta que un 7B rápido.

## Diferencia entre runtime y modelo

Un error común es pensar que “Ollama es peor/mejor que MLX” en abstracto. Lo correcto es comparar:

```text
mismo modelo + misma cuantización + mismo contexto + mismo prompt + runtime diferente
```

Solo así sabes si la diferencia viene del motor o del modelo elegido. Esta disciplina te servirá en [03-Cuantizacion](02-Cuantizacion-y-formatos.md) y [06-Proyecto-Final](../06-Proyectos/01-Asistente-local-completo.md).

## 1. Tres rutas de inferencia

En los tres sistemas conviene conocer estas herramientas:

- **Ollama**: la más simple para uso diario y API local.
- **llama.cpp**: control fino sobre GGUF, backend de GPU, batch y servidor.
- **LM Studio**: interfaz gráfica y servidor local para quien no quiera empezar por terminal.
- **MLX / mlx-lm**: ruta exclusiva de Apple Silicon y útil para [Fine-tuning con MLX](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

No necesitas casarte con una. Usa Ollama o LM Studio para empezar, llama.cpp para tener control y MLX para experimentar en el ecosistema Apple.

## 2. Ollama: uso básico

Instala y arranca Ollama como en [Instalación](../00-Introduccion/02-Instalacion-Windows-y-macOS.md). En Windows la app suele arrancar en segundo plano; en Mac puedes abrirla así:

```bash
open -a Ollama
```

Los comandos siguientes son iguales en PowerShell y macOS:

```text
ollama pull qwen3.5:4b
ollama run qwen3.5:4b "Escribe un checklist de 5 puntos para probar IA local."
ollama list
ollama show qwen3.5:4b
```

No ejecutes `ollama rm` hasta que quieras borrar de verdad el modelo. La primera práctica usa la API en [Tu primera IA local](../00-Introduccion/03-Tu-primera-IA-local.md).

Si 4B va lento, baja a 0,8B–2B. Si va holgado, compara después con 9B u otra familia.

## 3. Modelfile de Ollama

Crea un `Modelfile` con tu editor. Su contenido es idéntico en los tres sistemas:

```dockerfile
FROM qwen3.5:4b
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
SYSTEM "Eres un asistente técnico directo, preciso y práctico."
```

macOS:

```bash
ollama create qwen-tecnico -f ~/ia-local/laboratorio/Modelfile
ollama run qwen-tecnico "Dame un comando para ver la memoria."
```

Windows PowerShell:

```powershell
ollama create qwen-tecnico -f "$HOME\ia-local\laboratorio\Modelfile"
ollama run qwen-tecnico "Dame un comando para ver la memoria."
```

## 4. llama.cpp directo

Puedes dejar que llama.cpp descargue un GGUF de ejemplo directamente desde Hugging Face. Este comando funciona en Linux, Windows y macOS si instalaste el paquete:

```text
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Dame tres reglas para elegir un modelo local." -n 120 -c 4096
```

Cuando ya tengas un fichero propio:

```bash
llama-cli -m ~/ia-local/modelos/modelo.gguf \
  -p "Dame tres reglas para elegir un modelo local." \
  -n 200 -ngl 99 -c 4096
```

Windows PowerShell:

```powershell
llama-cli.exe `
  -m "$HOME\ia-local\modelos\modelo.gguf" `
  -p "Dame tres reglas para elegir un modelo local." `
  -n 200 -ngl 99 `
  -c 4096
```

Parámetros clave:

- `-m`: ruta al GGUF.
- `-ngl 99`: intenta mandar capas a la GPU disponible.
- `-c 4096`: contexto.
- `-n 200`: tokens máximos generados.

Si falla por memoria, reduce `-c`, usa Q4 en vez de Q5/Q6 o baja a un modelo 3B.

## 5. llama-server compatible con OpenAI

```bash
llama-server -m ~/ia-local/modelos/modelo.gguf \
  -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
```

Windows PowerShell:

```powershell
llama-server.exe -m "$HOME\ia-local\modelos\modelo.gguf" `
  -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
```

En otra terminal:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-qwen",
    "messages": [{"role":"user","content":"Resume las ventajas de GGUF."}],
    "temperature": 0.2
  }' | jq
```

En PowerShell, usa `Invoke-RestMethod` como en [Tu primera IA local](../00-Introduccion/03-Tu-primera-IA-local.md). Esto permite conectar apps que acepten una URL base compatible con OpenAI.

## 6. MLX / mlx-lm

MLX suele ir muy bien sobre Apple Silicon. Descarga y genera:

```bash
cd ~/ia-local/curso
source .venv/bin/activate
mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --prompt "Explica por qué MLX funciona bien en Apple Silicon." \
  --max-tokens 200 \
  --temp 0.3
```

Servidor MLX OpenAI-compatible:

```bash
mlx_lm.server \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8081
```

Prueba:

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen2.5-7B-Instruct-4bit","messages":[{"role":"user","content":"Di hola en español."}]}' | jq
```

## 7. Parámetros de generación

- `temperature`: creatividad. 0.1-0.3 para factual; 0.7+ para ideación.
- `top_p`: recorta el conjunto de tokens candidatos. 0.8-0.95 suele ser razonable.
- `repeat_penalty`: penaliza repeticiones. 1.05-1.2 suele bastar.
- `max_tokens`: longitud de salida.
- `num_ctx` / `-c`: ventana de contexto. Más no siempre es mejor: consume memoria.

Regla práctica:

```text
trabajo factual: temperature 0.2, top_p 0.9
brainstorming: temperature 0.7, top_p 0.95
código: temperature 0.1-0.3
```

## 8. Integración con agentes y MCP

Muchas herramientas aceptan una API compatible con OpenAI. En macOS:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="local-no-key"
```

En Windows PowerShell:

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:8080/v1"
$env:OPENAI_API_KEY = "local-no-key"
```

Para MCP o agentes locales, usa un modelo rápido de 3B-8B. Los agentes hacen muchas llamadas; un 14B local puede sentirse pesado.

## Profundización: prefill, decode y latencia percibida

La inferencia tiene dos fases importantes:

1. **Prefill**: el modelo procesa el prompt completo inicial.
2. **Decode**: genera tokens nuevos uno a uno.

Un prompt largo puede tardar mucho en la fase de prefill aunque luego el modelo genere rápido. Por eso, en agentes y chats largos, la latencia no depende solo de tokens/segundo de salida.

```text
prompt largo → prefill caro
respuesta larga → decode caro
contexto acumulado → KV cache grande
```

Cuando una app local “se siente lenta”, pregunta: ¿está tardando en leer el contexto o en generar la respuesta?

### Batch, contexto y throughput

`batch` controla cuántos tokens se procesan juntos en ciertas fases. Un batch grande puede mejorar throughput, pero también usar más memoria. En 24 GB, no conviene subir parámetros a ciegas.

El contexto (`num_ctx`, `-c`) es otra palanca crítica. Un contexto de 32k puede sonar atractivo, pero si tu tarea cabe en 4k-8k, el exceso solo aumenta consumo y latencia.

### Determinismo y creatividad

Para depurar, usa temperatura baja. Si cada ejecución cambia mucho, no sabes si una mejora viene del modelo, del prompt o del azar del muestreo.

```text
comparaciones técnicas → temperature 0.0-0.2
redacción creativa → temperature 0.7+
```

Cuando construyas evaluaciones en [06-Proyecto-Final](../06-Proyectos/01-Asistente-local-completo.md), usa parámetros estables. La creatividad es útil para escribir; es mala para medir.

### Servir una API local: qué cambia conceptualmente

Cuando levantas `llama-server`, `mlx_lm.server` u Ollama API, conviertes un modelo en un servicio. Eso añade nuevos problemas:

- concurrencia: varias peticiones a la vez;
- estado: cada petición debe traer su contexto;
- latencia: tiempo hasta primer token y tiempo total;
- límites: tamaño máximo de prompt/respuesta;
- integración: clientes esperan formato OpenAI-compatible.

Un servidor local no hace que el modelo sea más inteligente. Solo cambia cómo otras aplicaciones se comunican con él.

### Cómo comparar herramientas sin autoengañarte

Para comparar Ollama, llama.cpp y MLX, fija estas variables:

- mismo modelo o equivalente real;
- misma cuantización;
- mismo prompt;
- mismo contexto;
- misma temperatura;
- misma máquina y apps abiertas.

Si cambias varias cosas a la vez, no estás comparando runtimes, estás comparando configuraciones completas.

## Ejercicio práctico

1. Ejecuta el mismo prompt en Ollama, llama.cpp y MLX.
2. Mide subjetivamente:
   - velocidad;
   - calidad;
   - memoria;
   - facilidad de uso.
3. Levanta un servidor local y haz una llamada `/v1/chat/completions` con `curl`.
4. Documenta qué herramienta usarías para trabajo diario.

Prompt sugerido:

```text
Crea un plan de 7 días para aprender inferencia local en Mac, sin relleno.
```

## Recursos

- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- llama.cpp server: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- MLX LM: https://github.com/ml-explore/mlx-lm
- OpenAI-compatible API spec: https://platform.openai.com/docs/api-reference/chat

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Memoria, contexto y KV cache](../01-Fundamentos/03-Memoria-contexto-y-KV-cache-en-Apple-Silicon.md) · [Índice](../README.md) · [03 - Cuantización y reducción de memoria →](02-Cuantizacion-y-formatos.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
