---
title: Instalación en Linux, Windows y macOS
curso: IA-Local-de-Cero-a-Produccion
modulo: "00.02"
actualizado: 2026-07-24
---

# Prepara Linux, Windows o macOS

<!-- CURSO_NAV_TOP -->
[← Elige una ruta que encaje en tu equipo](01-Elige-hardware-y-modelo.md) · [Índice](../README.md) · [Tu primera sesión de IA local →](03-Tu-primera-IA-local.md)
<!-- /CURSO_NAV_TOP -->



Vamos a instalar una ruta sencilla y una ruta de desarrollo. No necesitas compilar nada para completar el primer bloque.

## Resultado esperado

Al terminar tendrás:

- Git y Python aislado con `uv`;
- Ollama funcionando en `localhost:11434`;
- LM Studio como alternativa gráfica;
- llama.cpp instalado para controlar modelos GGUF;
- una ruta nativa para Linux de escritorio o servidor;
- WSL2 si usas Windows y vas a hacer laboratorios CUDA;
- MLX si usas Apple Silicon y vas a seguir el itinerario de fine-tuning.

## 1. Herramientas básicas

### macOS con Apple Silicon

Instala las herramientas de compilación:

```bash
xcode-select --install
xcode-select -p
```

Instala [Homebrew](https://brew.sh/) si no lo tienes y después:

```bash
brew update
brew install git git-lfs uv python@3.12 cmake jq
git lfs install
```

Comprueba:

```bash
git --version
uv --version
python3 --version
cmake --version
```

### Windows PowerShell

Abre PowerShell normal, no hace falta ejecutarlo como administrador:

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id Python.Python.3.12 -e
winget install --id Kitware.CMake -e
git lfs install
```

Cierra y vuelve a abrir PowerShell para actualizar el `PATH`. Comprueba:

```powershell
git --version
uv --version
python --version
cmake --version
```

Si `winget` no está disponible, instala las mismas herramientas desde sus páginas oficiales. No descargues instaladores desde webs de terceros.

### Linux nativo (Ubuntu/Debian)

En otras distribuciones, sustituye `apt` por su gestor de paquetes e instala los mismos componentes:

```bash
sudo apt update
sudo apt install -y build-essential git git-lfs curl cmake jq python3 pciutils
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Cierra y abre la terminal si `uv` todavía no está en el `PATH`. Comprueba:

```bash
git --version
uv --version
python3 --version
cmake --version
```

Estos comandos sirven tanto en un escritorio Linux como en una sesión SSH de servidor. No necesitas interfaz gráfica para Ollama, llama.cpp ni los laboratorios Python.

## 2. Crea el espacio de prácticas

### macOS

```bash
mkdir -p ~/ia-local/laboratorio
cd ~/ia-local/laboratorio
uv venv --python 3.12
source .venv/bin/activate
uv pip install requests
python --version
```

### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force "$HOME\ia-local\laboratorio"
Set-Location "$HOME\ia-local\laboratorio"
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install requests
python --version
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Lee lo que te pregunta y cambia solo el ámbito `CurrentUser`. También puedes evitar activar el entorno y ejecutar `uv run python`.

### Linux

```bash
mkdir -p ~/ia-local/laboratorio
cd ~/ia-local/laboratorio
uv venv --python 3.12
source .venv/bin/activate
uv pip install requests
python --version
```

## 3. Instala Ollama

### macOS

Descarga la app desde [ollama.com/download](https://ollama.com/download) o usa:

```bash
brew install --cask ollama
open -a Ollama
```

### Windows

Descarga y ejecuta el instalador oficial desde [ollama.com/download/windows](https://ollama.com/download/windows). Ollama funciona como aplicación nativa en Windows 10 22H2 o posterior y deja disponible el comando en PowerShell.

### Linux

La instalación oficial para Linux es:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

En distribuciones con systemd, comprueba el servicio:

```bash
systemctl status ollama --no-pager
ollama --version
```

Si tu entorno no usa systemd —por ejemplo, algunos contenedores— inicia el proceso en una terminal con `ollama serve`. Para actualizar, vuelve a ejecutar el instalador oficial.

### Comprobación común

```text
ollama --version
ollama list
```

En macOS:

```bash
curl http://localhost:11434/api/tags
```

En Windows PowerShell:

```powershell
curl.exe http://localhost:11434/api/tags
```

En Linux:

```bash
curl http://localhost:11434/api/tags
```

Debes recibir JSON. Una lista vacía es correcta si aún no has descargado modelos.

## 4. Instala LM Studio

Descárgalo desde [lmstudio.ai](https://lmstudio.ai/). Es la ruta visual: permite buscar modelos, ver si caben, conversar y levantar un servidor local.

Requisitos oficiales a fecha de este curso:

- macOS 14 o posterior con Apple Silicon; 16 GB recomendados;
- Windows x64 con AVX2 o Windows ARM; 16 GB RAM y 4 GB VRAM recomendados.
- Linux x64 o ARM64, con Ubuntu 20.04 o posterior como base soportada; se distribuye como AppImage y requiere un entorno gráfico compatible.

Con 8 GB aún puedes practicar con modelos pequeños y contexto corto. No descargues un modelo solo porque la interfaz lo muestra: mira tamaño, cuantización y memoria estimada. En un servidor Linux sin escritorio, omite LM Studio y usa Ollama o llama.cpp por terminal.

## 5. Instala llama.cpp sin compilar

### macOS

```bash
brew install llama.cpp
llama-cli --help
llama-server --help
```

Metal viene activado por defecto en la compilación oficial.

### Windows

```powershell
winget install llama.cpp
llama-cli --help
llama-server --help
```

Si quieres un backend concreto (CUDA, Vulkan, HIP o SYCL), usa los binarios de la [página oficial de releases](https://github.com/ggml-org/llama.cpp/releases) o sigue la [guía de compilación](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md). No necesitas hacerlo aún.

### Linux

La ruta universal es compilar desde el repositorio oficial. Para CPU:

```bash
cd ~/ia-local
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release -j
./build/bin/llama-cli --help
./build/bin/llama-server --help
```

Con un toolkit CUDA ya funcional, cambia la configuración por `cmake -B build -DGGML_CUDA=ON`. Para AMD, Vulkan, Intel u otros backends, sigue la [guía oficial de compilación](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) y verifica primero drivers y compatibilidad.

## 6. Solo Apple Silicon: MLX

Activa el entorno del laboratorio:

```bash
cd ~/ia-local/laboratorio
source .venv/bin/activate
uv pip install mlx mlx-lm
python -c "import mlx.core as mx; print(mx.default_device())"
mlx_lm.generate --help
```

MLX no es la ruta de Windows ni Linux. En Linux con una GPU compatible usarás PyTorch/PEFT de forma nativa; en Windows con NVIDIA, desde WSL2.

## 7. Solo Windows avanzado: WSL2

WSL2 te da un Linux integrado. Lo usaremos para herramientas cuyo camino principal es Linux/CUDA.

En PowerShell como administrador:

```powershell
wsl --install -d Ubuntu
wsl --update
wsl --status
```

Reinicia cuando Windows lo pida. Dentro de Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential git git-lfs curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

No instales un driver NVIDIA Linux dentro de WSL si ya sigues la ruta oficial de NVIDIA para WSL; el driver del host expone la GPU. Comprueba dentro de WSL:

```bash
nvidia-smi
```

Si no tienes NVIDIA, WSL sigue siendo útil para reproducir comandos Linux, pero no crea una GPU de la nada.

## 8. Foto del entorno

Guarda estas versiones en `MI-LABORATORIO.md`:

### macOS

```bash
sw_vers
git --version
uv --version
ollama --version
llama-cli --version
```

### Windows PowerShell

```powershell
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsArchitecture
git --version
uv --version
ollama --version
llama-cli --version
nvidia-smi
```

### Linux

```bash
cat /etc/os-release
uname -m
free -h
git --version
uv --version
ollama --version
./build/bin/llama-cli --version  # desde el directorio de llama.cpp
nvidia-smi                       # solo NVIDIA
```

## Problemas típicos

| Lo que ves | Qué hacer |
|---|---|
| El comando no aparece después de instalar | Cierra y abre la terminal. |
| Ollama no responde | Abre la aplicación. Si usas el binario sin app, ejecuta `ollama serve`. |
| El modelo va a descargarse en un disco lleno | Configura `OLLAMA_MODELS` antes de descargar. |
| Windows muestra rutas distintas | Consulta la [documentación oficial de Ollama para Windows](https://docs.ollama.com/windows). |
| Has mezclado PowerShell y WSL | Decide en qué terminal estás. Las rutas de Windows y `$HOME` en WSL pertenecen a sistemas distintos. |
| Linux instaló Ollama pero no responde | Revisa `systemctl status ollama` y `journalctl -u ollama -n 50 --no-pager`. |
| El servidor no tiene escritorio | Omite LM Studio; usa Ollama o llama.cpp por SSH y conserva la API en `127.0.0.1`. |

## Criterio de salida

No avances hasta que:

- `ollama list` responda;
- la API de Ollama devuelva JSON;
- `llama-cli --help` funcione;
- sepas activar tu entorno Python o usar `uv run`;
- puedas decir qué herramientas son nativas y cuáles dependen de tu plataforma.

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Elige una ruta que encaje en tu equipo](01-Elige-hardware-y-modelo.md) · [Índice](../README.md) · [Tu primera sesión de IA local →](03-Tu-primera-IA-local.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
