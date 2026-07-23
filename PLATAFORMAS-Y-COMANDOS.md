---
title: Plataformas y comandos
curso: IA-Local-de-Cero-a-Produccion
---

# Equivalencias entre Windows y macOS

<!-- CURSO_NAV_TOP -->
[← Elige tu itinerario y comprueba el aprendizaje](00-Introduccion/04-Rutas-y-autoevaluacion.md) · [Índice](README.md) · [Fundamentos: qué es un LLM y qué necesita tu equipo →](01-Fundamentos/01-Que-es-un-LLM.md)
<!-- /CURSO_NAV_TOP -->



Esta página evita repetir diferencias pequeñas en todos los capítulos. Cuando un comando sea idéntico, verás un único bloque `text`. Cuando cambie, usa la columna de tu sistema.

## Qué terminal estás usando

| Plataforma | Terminal principal | Ruta típica |
|---|---|---|
| macOS | Terminal o iTerm, shell zsh | `$HOME/ia-local` |
| Windows nativo | PowerShell | `$HOME\ia-local` |
| Windows avanzado | Ubuntu en WSL2, shell bash | `$HOME/ia-local` |

WSL2 no es PowerShell. Un entorno virtual creado en Windows no debe reutilizarse desde WSL, ni al revés.

## Moverte y crear carpetas

| Acción | macOS / WSL2 | Windows PowerShell |
|---|---|---|
| carpeta actual | `pwd` | `Get-Location` |
| listar | `ls -la` | `Get-ChildItem -Force` |
| crear carpeta | `mkdir -p ia-local` | `New-Item -ItemType Directory -Force ia-local` |
| entrar | `cd ia-local` | `Set-Location ia-local` |
| borrar un fichero | `rm fichero.txt` | `Remove-Item fichero.txt` |

Evita copiar comandos de borrado recursivo si no entiendes exactamente la ruta.

## Entorno Python con uv

Crear:

```text
uv venv --python 3.12
```

Activar en macOS/WSL2:

```bash
source .venv/bin/activate
```

Activar en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Ruta que evita activar:

```text
uv run python programa.py
uv run pytest
```

## Variables de entorno

macOS/WSL2, solo para la terminal actual:

```bash
export MI_VARIABLE="valor"
echo "$MI_VARIABLE"
```

Windows PowerShell, solo para la terminal actual:

```powershell
$env:MI_VARIABLE = "valor"
$env:MI_VARIABLE
```

No pegues tokens o contraseñas en capturas, commits ni cuadernos del curso.

## HTTP y JSON

En macOS/WSL2, `curl` es el binario habitual. En PowerShell usa `curl.exe` si quieres copiar una orden curl literalmente; `curl` puede ser un alias distinto según la versión.

Para JSON complejo, PowerShell suele ser más claro con `Invoke-RestMethod` y `ConvertTo-Json`. En Python, `requests` funciona igual en ambos sistemas.

## Runtimes y aceleración

| Herramienta | macOS Apple Silicon | Windows |
|---|---|---|
| Ollama | Metal | NVIDIA y AMD compatibles; CPU como fallback |
| LM Studio | Apple Silicon/Metal | x64 o ARM; GPU compatible o CPU |
| llama.cpp | Metal por defecto | CUDA, Vulkan, HIP, SYCL o CPU |
| MLX / mlx-lm | sí, Apple Silicon | no es la ruta de Windows |
| PyTorch + PEFT | MPS con limitaciones según técnica | CUDA en WSL2 recomendado |
| vLLM | vLLM-Metal para Apple Silicon | WSL2/Linux para CUDA y otros backends |

## llama.cpp: mismo servidor, distinto ejecutable

macOS:

```bash
llama-server -m modelos/modelo.gguf -c 4096 --host 127.0.0.1 --port 8080
```

Windows PowerShell:

```powershell
llama-server.exe -m modelos\modelo.gguf -c 4096 --host 127.0.0.1 --port 8080
```

Si `winget` instaló el ejecutable sin sufijo visible en tu terminal, `llama-server` también puede funcionar. Compruébalo con `Get-Command llama-server*`.

## Monitorización

macOS:

```bash
memory_pressure
vm_stat
top -l 1 | head -n 20
```

Windows PowerShell:

```powershell
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 Name, CPU, WorkingSet64
Get-Counter '\Memory\Available MBytes'
nvidia-smi
```

WSL2 con NVIDIA:

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## Qué camino seguir en capítulos antiguos centrados en Mac

- Sustituye activación de `.venv` por la equivalencia PowerShell.
- Las órdenes `ollama`, `git`, `uv` y la mayoría de scripts Python no cambian.
- Cuando aparezca MLX, sigue [Fine-tuning con PEFT y QLoRA](04-Adaptar/02-Fine-tuning-con-PEFT-y-QLoRA.md) si tienes NVIDIA.
- Cuando aparezca Metal, usa CUDA o Vulkan en llama.cpp.
- Si el laboratorio necesita vLLM/CUDA, ejecútalo en WSL2, Linux o cloud.

## Fuentes oficiales

- [Ollama para Windows](https://docs.ollama.com/windows)
- [Instalación de llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)
- [Compilación y backends de llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [Requisitos de LM Studio](https://lmstudio.ai/docs/app/system-requirements)
- [vLLM quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Elige tu itinerario y comprueba el aprendizaje](00-Introduccion/04-Rutas-y-autoevaluacion.md) · [Índice](README.md) · [Fundamentos: qué es un LLM y qué necesita tu equipo →](01-Fundamentos/01-Que-es-un-LLM.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
