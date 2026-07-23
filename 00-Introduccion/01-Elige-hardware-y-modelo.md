---
title: Elige hardware y modelo
curso: IA-Local-de-Cero-a-Produccion
modulo: "00.01"
actualizado: 2026-07-23
---

# Elige una ruta que encaje en tu equipo

<!-- CURSO_NAV_TOP -->
[← Bienvenida: cómo aprender IA local sin profesor](00-Bienvenida-y-metodo.md) · [Índice](../README.md) · [Prepara Linux, Windows o macOS →](02-Instalacion-Windows-y-macOS.md)
<!-- /CURSO_NAV_TOP -->



Busca un modelo que quepa, responda a una velocidad aceptable y resuelva tu tarea. El que encabeza un ranking puede ser una mala elección para tu equipo.

## 1. Haz inventario

### macOS

```bash
sw_vers
uname -m
system_profiler SPHardwareDataType
df -h /
```

En un Mac con Apple Silicon, CPU y GPU comparten **memoria unificada**. Un Mac de 24 GB no tiene "24 GB para el modelo": macOS, las aplicaciones, los pesos y la KV cache compiten por el mismo espacio.

### Windows PowerShell

```powershell
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsArchitecture, CsTotalPhysicalMemory
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion
Get-PSDrive C
nvidia-smi
```

Si `nvidia-smi` no existe, puede que no tengas NVIDIA o que falte el driver. En Windows con GPU dedicada hay dos presupuestos: **VRAM** de la tarjeta y **RAM** del sistema. Si parte del modelo cae a CPU porque no cabe en VRAM, funcionará, pero normalmente irá más lento.

### Linux

```bash
cat /etc/os-release
uname -m
lscpu | sed -n '1,12p'
free -h
df -h /
lspci | grep -Ei 'vga|3d|display'
nvidia-smi
```

`nvidia-smi` solo aplica a NVIDIA. En un servidor Linux, comprueba además que la GPU esté visible dentro del contenedor o de la sesión remota antes de atribuir un fallo al runtime. La regla sigue siendo la misma: separa RAM del sistema y VRAM, y reserva margen para el sistema, los buffers y otras cargas.

## 2. Regla de memoria que no promete milagros

Para un modelo denso:

```text
pesos aproximados = parámetros × bits_por_peso / 8
memoria real = pesos + KV cache + buffers del runtime + sistema y aplicaciones
```

Un 7B a 4 bits ocupa unos 3,5 GB solo en teoría. El fichero GGUF y la ejecución real necesitan más. Deja margen:

- en memoria unificada, intenta no pasar del 70-75 % con pesos y contexto;
- en una GPU dedicada, deja al menos 1-2 GB de VRAM para buffers y entorno gráfico;
- empieza con contexto 4k u 8k, no con el máximo anunciado por el modelo;
- cierra juegos, editores de vídeo y otras cargas de GPU antes de medir.

## 3. Ruta conservadora por memoria disponible

Los nombres concretos cambian rápido; los tamaños son el criterio estable. Esta tabla usa cuantización alrededor de Q4 y contexto moderado.

| Memoria disponible | Empieza con | Tamaño orientativo | Qué puedes hacer |
|---|---|---:|---|
| 8 GB RAM, sin GPU útil | Qwen 3.5 0.8B/2B, Gemma 3 1B | 0,8B-2B | aprender CLI, prompts, API y embeddings pequeños |
| 16 GB RAM o 6-8 GB VRAM | Qwen 3.5 4B, Gemma 3 4B, Phi-4-mini | 3B-4B | chat, RAG, tool calling sencillo, código ligero |
| 24-32 GB unificados o 12-16 GB VRAM | Qwen 3.5 9B, Gemma 3 12B, Qwen3 8B/14B | 8B-14B | uso general sólido, visión según modelo, LoRA pequeño |
| 48-64 GB unificados o 24 GB VRAM | Qwen 3.5 27B, Gemma 3 27B u otros 20B-32B Q4 | 20B-32B | más calidad, contexto mayor, serving con poca concurrencia |
| varias GPUs o 80 GB+ | modelos grandes y MoE | 70B+ o MoE | laboratorios de producción y paralelismo |

> [!warning] "Cabe" no significa "va bien"
> Un modelo que deja el equipo haciendo swap o que genera a menos de un token por segundo no es una elección práctica. Baja un tamaño o reduce contexto.

## 4. Modelos con papeles distintos

Prueba al menos tres familias, no tres tamaños del mismo modelo:

- Qwen 3.5 es multilingüe, multimodal y admite tool use. Sus tamaños empiezan en 0.8B, así que sirve para comparar cuánto cambia una familia al crecer.
- Gemma 3 y Gemma 4 ofrecen tamaños pequeños y opciones multimodales. Revisa la licencia y los tags disponibles.
- Phi-4-mini es compacto y permite estudiar el equilibrio entre tamaño y capacidad.
- Qwen3-0.6B es el modelo ancla de LLMOps. Lo usamos porque deja inspeccionar arquitectura e inferencia con poco hardware, no porque sea el mejor chatbot.
- También necesitarás un modelo de embeddings. No conversa: transforma texto en vectores para RAG. `nomic-embed-text` es una opción sencilla en Ollama.

La [biblioteca oficial de Ollama](https://ollama.com/search) muestra los tags que existen hoy. No inventes un nombre: abre la ficha y copia el comando exacto.

## 5. El selector rápido

```text
¿Solo quieres empezar hoy?
  sí → Ollama + modelo de 1B-4B
  no
   ├─ ¿prefieres interfaz gráfica? → LM Studio
   ├─ ¿quieres controlar GGUF y flags? → llama.cpp
   ├─ ¿quieres fine-tuning en Mac? → MLX
   └─ ¿quieres CUDA/serving avanzado? → Linux nativo o WSL2 + PEFT/vLLM
```

## 6. Licencia, model card y datos

"Open weights" no siempre significa "open source" ni permite cualquier uso comercial. Antes de elegir un modelo para algo que saldrá del laboratorio:

1. abre su model card oficial;
2. identifica la licencia exacta;
3. comprueba límites de uso y distribución;
4. revisa idiomas, contexto, formato de chat y casos no recomendados;
5. guarda el identificador y la revisión del modelo.

## Comprobación

Antes de pasar a instalación deberías poder completar:

```text
Mi equipo tiene:
Mi límite principal es:
Empezaré con el modelo:
Lo ejecutaré con:
He dejado este margen de memoria:
```

Fuentes de referencia: [requisitos de LM Studio](https://lmstudio.ai/docs/app/system-requirements), [modelos de Ollama](https://ollama.com/search) y [documentación de llama.cpp](https://github.com/ggml-org/llama.cpp).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Bienvenida: cómo aprender IA local sin profesor](00-Bienvenida-y-metodo.md) · [Índice](../README.md) · [Prepara Linux, Windows o macOS →](02-Instalacion-Windows-y-macOS.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
