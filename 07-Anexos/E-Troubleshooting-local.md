---
tags:
  - curso/ia-local
  - troubleshooting
  - errores
curso: IA-Local-de-Cero-a-Produccion
modulo: anexo-troubleshooting
estado: completo
---

# Troubleshooting: errores comunes y soluciones

<!-- CURSO_NAV_TOP -->
[← Modelos recomendados para Mac M2 24 GB](D-Modelos-para-Apple-Silicon-24GB.md) · [Índice](../README.md) · [Apéndice A - Fundamentos matemáticos →](F-Fundamentos-matematicos.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Linux, Windows y macOS
> Ollama, llama.cpp y Python sirven en los tres sistemas. MLX y Metal son exclusivos de Apple Silicon; en Linux usa CUDA, ROCm/HIP, Vulkan o CPU según tu equipo. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!goals]
> **Objetivos de este anexo:**
> - Diagnosticar y resolver los errores más frecuentes en Linux, Windows y macOS.
> - Cubrir seis categorías: memoria, Ollama, llama.cpp, MLX, Python/uv y formato de datos.
> - Para cada error: síntoma, causa, solución y comando de verificación.
> - Mínimo 20 errores documentados con soluciones concretas y ejecutables.

---


---

## Cómo usar este anexo

Cada error sigue el formato:

- **Síntoma:** qué ves (mensaje de error o comportamiento).
- **Causa:** por qué ocurre.
- **Solución:** pasos concretos.
- **Verificación:** comando para confirmar que se resolvió.

---

## Diagnóstico rápido por plataforma

Antes de cambiar dependencias o reinstalar nada, recoge evidencia.

### macOS

```bash
sw_vers
uname -m
memory_pressure
ollama --version
ollama ps
curl http://127.0.0.1:11434/api/tags
```

### Windows PowerShell

```powershell
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsArchitecture
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
Get-Counter '\Memory\Available MBytes'
nvidia-smi
ollama --version
ollama ps
curl.exe http://127.0.0.1:11434/api/tags
```

### Linux

```bash
cat /etc/os-release
uname -m
free -h
lspci | grep -Ei 'vga|3d|display'
nvidia-smi
systemctl status ollama --no-pager
ollama ps
curl http://127.0.0.1:11434/api/tags
```

Si es un servidor remoto, ejecuta estas comprobaciones dentro de su sesión SSH, no en el ordenador desde el que te conectas.

Que `nvidia-smi` falle no implica que Ollama esté roto: quizá no tienes NVIDIA. Mira el nombre de la GPU y usa el backend compatible.

### Windows: la GPU aparece en el host, pero no en WSL2

Dentro de Ubuntu:

```bash
nvidia-smi
uname -a
```

Si no aparece, vuelve a PowerShell, ejecuta `wsl --update`, actualiza el driver NVIDIA del host y reinicia. No instales a ciegas un driver de pantalla Linux dentro de WSL2.

### Windows: PowerShell bloquea el entorno virtual

Puedes evitar la activación:

```powershell
uv run python programa.py
```

O permitir scripts firmados o locales solo para tu usuario:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Windows: puerto 11434 ocupado

```powershell
Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue
Get-Process -Id (Get-NetTCPConnection -LocalPort 11434).OwningProcess
```

Normalmente significa que la app de Ollama ya está ejecutándose. No lances otra instancia.

---

## 1. Memoria (RAM, VRAM o memoria unificada)

### Error 1.1 — `malloc: failed to allocate` / `Out of memory`

- **Síntoma:** El proceso muere con un error de memoria al cargar el modelo.
- **Causa:** El modelo cuantizado + contexto excede la RAM disponible.
- **Solución:**
  1. Bajar la cuantización (Q8 → Q5 → Q4).
  2. Reducir `-c` (context size): `--ctx-size 4096` en vez de 8192.
  3. Cerrar navegadores y otras apps que consuman RAM.
  4. Usar un modelo más pequeño (7B en vez de 13B).
- **Verificación:**
  ```bash
  # Ver uso de memoria antes y durante
  top -l 1 -s 0 | grep PhysMem
  # Ver presión de memoria
  memory_pressure
  ```

### Error 1.2 — Modelo carga pero inference es lentísima (< 1 tok/s)

- **Síntoma:** Generación extremadamente lenta tras carga correcta.
- **Causa:** Memoria al límite → swapping a disco, lo que degrada el rendimiento 10–50×.
- **Solución:**
  1. Verificar que no hay swap activo (ver comando).
  2. Reducir modelo o contexto.
  3. En Ollama, bajar `num_ctx` en el Modelfile.
- **Verificación:**
  ```bash
  # Ver uso de swap
  sysctl vm.swapusage
  # Si "used" > 0 durante inference, hay swapping
  ```

### Error 1.3 — `Killed: 9` al ejecutar modelo

- **Síntoma:** El proceso termina con signal 9 (SIGKILL) sin mensaje.
- **Causa:** macOS mata procesos que exceden el límite de memoria.
- **Solución:**
  1. Reducir tamaño de modelo / contexto.
  2. Aumentar el límite de descriptors si aplica: `ulimit -n 65536`.
  3. Verificar que no hay fugas en el código Python.
- **Verificación:**
  ```bash
  # Ver logs del sistema
  log show --predicate 'eventMessage contains "Jetsam"' --last 5m
  ```

---

## 2. Ollama

### Error 2.1 — `connection refused` en localhost:11434

- **Síntoma:** `curl http://localhost:11434/api/tags` falla.
- **Causa:** El servicio de Ollama no está corriendo.
- **Solución:**
  ```bash
  # macOS con Homebrew
  brew services start ollama

  # Linux con systemd
  sudo systemctl start ollama
  systemctl status ollama --no-pager

  # Cualquier sistema Unix: lanzar manualmente
  ollama serve &
  ```
- **Verificación:**
  ```bash
  curl -s http://localhost:11434/api/tags | python3 -m json.tool
  ```

### Error 2.2 — `Error: model not found` al hacer `ollama run`

- **Símtoma:** `ollama run llama3.1:8b` devuelve error de modelo no encontrado.
- **Causa:** El modelo no está descargado localmente.
- **Solución:**
  ```bash
  ollama pull llama3.1:8b
  ollama list
  ```
- **Verificación:**
  ```bash
  ollama list | grep llama3.1
  ```

### Error 2.3 — `ollama serve` falla por puerto ocupado

- **Síntoma:** `listen tcp 127.0.0.1:11434: bind: address already in use`.
- **Causa:** Hay otra instancia de Ollama corriendo.
- **Solución:**
  ```bash
  # Ver qué proceso ocupa el puerto
  lsof -i :11434
  # Alternativa habitual en Linux
  ss -ltnp | grep 11434
  # Matar el proceso
  kill -9 <PID>
  # O usar otro puerto
  OLLAMA_HOST=127.0.0.1:11435 ollama serve &
  ```
- **Verificación:**
  ```bash
  lsof -i :11434
  ```

### Error 2.4 — Respuesta truncada / contexto insuficiente

- **Síntoma:** El modelo corta la respuesta a media frase o no sigue instrucciones largas.
- **Causa:** el contexto configurado —4096 en muchas instalaciones por defecto— puede ser insuficiente, o el límite de salida puede cortar la respuesta.
- **Solución:** Crear Modelfile personalizado:
  ```bash
  # Crear Modelfile
  cat > Modelfile <<EOF
  FROM llama3.1:8b
  PARAMETER num_ctx 8192
  PARAMETER temperature 0.7
  EOF
  ollama create llama3.1-8b-8k -f Modelfile
  ollama run llama3.1-8b-8k
  ```
- **Verificación:**
  ```bash
  ollama show llama3.1-8b-8k --modelfile | grep num_ctx
  ```

---

## 3. llama.cpp

### Error 3.1 — `error: no Metal device found`

- **Síntoma:** Al ejecutar `./llama-cli` se queja de que no hay GPU Metal.
- **Causa:** Compilado sin `GGML_METAL=1` o Xcode Command Line Tools no instalado.
- **Solución:**
  ```bash
  # Instalar CLT
  xcode-select --install
  # Recompilar; Metal se activa por defecto en Apple Silicon
  cd ~/proyectos/llama.cpp
  cmake -B build
  cmake --build build --config Release -j
  ```
- **Verificación:**
  ```bash
  ./llama-cli -m models/test.gguf -p "test" -n 1 2>&1 | grep -i metal
  # Debe mostrar: ggml_metal_init: allocating
  ```

### Error 3.2 — `error: model file format version x not supported`

- **Síntoma:** Error al cargar un archivo GGUF.
- **Causa:** El GGUF fue creado con una versión más nueva de llama.cpp que la instalada.
- **Solución:**
  ```bash
  cd ~/proyectos/llama.cpp
  git pull
  cmake -B build
  cmake --build build --config Release -j
  ```
- **Verificación:**
  ```bash
  ./llama-cli --version
  ```

### Error 3.3 — `failed to load model: model size too large for context`

- **Síntoma:** Error al cargar el modelo indicando que no cabe en contexto/KV cache.
- **Causa:** `--ctx-size` es demasiado grande para la RAM disponible.
- **Solución:**
  ```bash
  # Reducir contexto
  ./llama-cli -m model.gguf -c 4096 -n 512
  # O usar KV cache cuantizada (Flash Attention)
  ./llama-cli -m model.gguf -c 8192 -fa -n 512
  ```
- **Verificación:**
  ```bash
  ./llama-cli -m model.gguf -c 4096 -n 10 -p "Hola" 2>&1 | tail -20
  ```

### Error 3.4 — Modelos multimodales no procesan imágenes

- **Síntoma:** Qwen2-VL / LLaVA devuelve texto pero ignora la imagen (`--image` no funciona).
- **Causa:** Falta el archivo `mmproj-*.gguf` (vision projector).
- **Solución:**
  ```bash
  # Descarga el modelo y el mmproj indicados por la ficha del GGUF.
  # Ejecuta con ambos; los nombres dependen del modelo elegido.
  ./llama-cli -m model.gguf --mmproj mmproj.gguf \
    --image foto.jpg -p "Describe" -n 256
  ```
- **Verificación:**
  ```bash
  ls -lh models/qwen2vl-7b/mmproj-*.gguf
  ```

---

## 4. MLX

### Error 4.1 — `ModuleNotFoundError: No module named 'mlx'`

- **Síntoma:** Python no encuentra mlx al importar.
- **Causa:** MLX no instalado o instalado en otro entorno.
- **Solución:**
  ```bash
  # Crear venv con uv
  uv venv && source .venv/bin/activate
  uv pip install mlx mlx-lm
  ```
- **Verificación:**
  ```bash
  python -c "import mlx; print(mlx.__version__)"
  ```

### Error 4.2 — `RuntimeError: Unsupported device` o sin GPU

- **Símtoma:** MLX se queja del dispositivo.
- **Causa:** Estás corriendo en una Mac sin Apple Silicon o en modo Rosetta.
- **Solución:**
  ```bash
  # Verificar arquitectura
  uname -m  # debe ser arm64
  # Asegurar que Python es arm64
  python -c "import platform; print(platform.machine())"
  # Si dice x86_64, reinstalar Python arm64
  brew install python@3.12
  ```
- **Verificación:**
  ```bash
  python -c "import mlx.core as mx; print(mx.default_device())"
  # Debe mostrar Device(gpu, 0)
  ```

### Error 4.3 — Modelo descargado desde HuggingFace no carga en MLX

- **Síntoma:** `ValueError` al cargar modelo con `mlx_lm.load`.
- **Causa:** El modelo no está en formato MLX (necesita conversión o usar versión `mlx-community/...`).
- **Solución:**
  ```bash
  # Buscar versión MLX en HuggingFace
  # Usar siempre modelos del namespace mlx-community
  mlx_lm.generate --model mlx-community/Llama-3.1-8B-Instruct-4bit \
    --prompt "Hola" --max-tokens 50
  ```
- **Verificación:**
  ```bash
  # Confirmar que el repo tiene config.json con formato MLX
  huggingface-cli download mlx-community/Llama-3.1-8B-Instruct-4bit \
    --local-dir ./test_model
  cat test_model/config.json | head -20
  ```

---

## 5. Python / uv

### Error 5.1 — `error: can not find a virtual environment`

- **Síntoma:** `uv pip install ...` falla sin venv.
- **Causa:** uv requiere un entorno virtual explícito.
- **Solución:**
  ```bash
  uv venv
  source .venv/bin/activate
  uv pip install <paquete>
  # O usar uv run directamente
  uv run python script.py
  ```
- **Verificación:**
  ```bash
  which python  # debe apuntar a .venv/bin/python
  ```

### Error 5.2 — `externally-managed-environment` al hacer `pip install`

- **Síntoma:** PEP 668 bloquea el pip global.
- **Causa:** Python del sistema marcado como externally managed.
- **Solución:**
  ```bash
  # No usar pip global. Usar uv o venv
  uv venv && source .venv/bin/activate
  uv pip install <paquete>
  # O instalar herramienta global con uv tool
  uv tool install <paquete>
  ```
- **Verificación:**
  ```bash
  uv pip list
  ```

### Error 5.3 — `Command not found: uv`

- **Símtoma:** uv no está en PATH.
- **Causa:** uv no instalado o PATH no configurado.
- **Solución:**
  ```bash
  # Instalar uv
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Recargar shell
  source ~/.zshrc
  # O añadir manualmente
  export PATH="$HOME/.local/bin:$PATH"
  ```
- **Verificación:**
  ```bash
  uv --version
  ```

### Error 5.4 — Dependencias en conflicto / `ResolutionImpossibleError`

- **Síntoma:** uv no puede resolver versiones compatibles de paquetes.
- **Causa:** Restricciones de versiones en conflicto en `pyproject.toml`.
- **Solución:**
  ```bash
  # Ver árbol de dependencias
  uv pip tree
  # Forzar versión específica
  uv pip install "numpy<2.0" "torch>=2.2"
  # O recrear entorno limpio
  rm -rf .venv uv.lock
  uv venv && uv pip install -r requirements.txt
  ```
- **Verificación:**
  ```bash
  uv pip list
  uv pip check
  ```

### Error 5.5 — `huggingface-cli: command not found`

- **Síntoma:** No puedes descargar modelos desde HuggingFace.
- **Causa:** `huggingface_hub[cli]` no instalado.
- **Solución:**
  ```bash
  uv tool install "huggingface_hub[cli]"
  # O en venv
  uv pip install "huggingface_hub[cli]"
  ```
- **Verificación:**
  ```bash
  huggingface-cli --version
  ```

---

## 6. Formato de datos / archivos

### Error 6.1 — whisper.cpp: `error: invalid audio file`

- **Síntoma:** whisper.cpp no carga el WAV.
- **Causa:** Formato no soportado (no es 16 kHz mono WAV).
- **Solución:**
  ```bash
  # Convertir con ffmpeg
  ffmpeg -i entrada.mp3 -ar 16000 -ac 1 -c:a pcm_s16le salida.wav
  ./build/bin/whisper-cli -m models/ggml-base.bin -f salida.wav
  ```
- **Verificación:**
  ```bash
  ffprobe salida.wav 2>&1 | grep -E "Sample rate|Channels|codec"
  # Debe mostrar 16000 Hz, mono, pcm_s16le
  ```

### Error 6.2 — `ffmpeg: command not found`

- **Síntoma:** No puedes convertir/procesar audio.
- **Causa:** ffmpeg no instalado.
- **Solución:**
  ```bash
  brew install ffmpeg
  ```
- **Verificación:**
  ```bash
  ffmpeg -version | head -1
  ```

### Error 6.3 — Salida JSON malformada al parsear respuesta de Ollama

- **Síntoma:** `json.decoder.JSONDecodeError` al leer respuesta de la API.
- **Causa:** Respuesta streaming o el modelo generó texto que no es JSON válido.
- **Solución:**

```python
import json
import requests

# Usar stream=False en la petición
resp = requests.post(
    url,
    json={"model": model, "prompt": prompt, "stream": False},
)

# O parsear línea a línea si stream=True
for line in resp.iter_lines():
    if line:
        chunk = json.loads(line)
```
- **Verificación:**
  ```bash
  curl -s http://localhost:11434/api/generate -d '{
    "model": "llama3.1:8b",
    "prompt": "test",
    "stream": false
  }' | python3 -m json.tool
  ```

### Error 6.4 — GGUF corrupto o incompleto (hash mismatch)

- **Síntoma:** `llama-cli` falla al cargar con `invalid magic number`.
- **Causa:** Descarga interrumpida o archivo corrupto.
- **Solución:**
  ```bash
  # Re-descargar
  rm models/model.gguf
  huggingface-cli download <repo> <file> --local-dir ./models
  # Verificar tamaño
  ls -lh models/model.gguf
  ```
- **Verificación:**
  ```bash
  # Comparar con tamaño esperado en HF
  md5 models/model.gguf
  ```

### Error 6.5 — Imagen no procesada por VLM (formato no soportado)

- **Síntoma:** Qwen2-VL / LLaVA ignora imagen o devuelve error.
- **Causa:** Formato no soportado (HEIC, WebP raro, o base64 corrupto).
- **Solución:**
  ```bash
  # Convertir a PNG/JPG estándar
  sips -s format png imagen.heic --out imagen.png
  # Verificar
  file imagen.png
  # Re-enviar
  IMG_B64=$(base64 -i imagen.png)
  ```
- **Verificación:**
  ```bash
  file imagen.png  # debe mostrar PNG image data
  ```

### Error 6.6 — `FileNotFoundError` al cargar dataset/archivo

- **Síntoma:** Python no encuentra el archivo especificado en el script.
- **Causa:** Ruta relativa mal resuelta o archivo en otra ubicación.
- **Solución:**
  ```python
  from pathlib import Path
  # Usar rutas absolutas o relativas al script
  BASE = Path(__file__).parent
  archivo = BASE / "data" / "input.txt"
  # Verificar existencia
  assert archivo.exists(), f"No existe: {archivo}"
  ```
- **Verificación:**
  ```bash
  ls -lh data/input.txt
  python -c "from pathlib import Path; print(Path('data/input.txt').resolve())"
  ```

---

## Resumen rápido: diagnóstico en 60 segundos

```bash
# 1. Estado del sistema
uname -m                              # arm64 esperado
sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}'
vm.swapusage                          # swap usage
top -l 1 -s 0 | grep PhysMem          # memoria libre

# 2. Servicios
brew services list | grep -E "ollama|whisper"
curl -s http://localhost:11434/api/tags | python3 -m json.tool

# 3. Toolchain
python3 --version
uv --version
huggingface-cli --version
ffmpeg -version | head -1

# 4. Compilaciones
ls -lh ~/proyectos/llama.cpp/llama-cli
ls -lh ~/proyectos/whisper.cpp/main

# 5. Modelos descargados
ollama list
ls -lh ~/proyectos/llama.cpp/models/
ls -lh ~/proyectos/whisper.cpp/models/
```

---

## Ejercicio práctico

> [!exercise]
> **Diagnóstico guiado**
>
> Ejecuta el bloque de "diagnóstico en 60 segundos" de arriba y guarda la salida:
>
> ```bash
> mkdir -p ejercicios/troubleshooting
> {
>   echo "## Diagnóstico del sistema"
>   echo "### Fecha: $(date)"
>   echo ""
>   echo "### Arquitectura y RAM"
>   uname -m
>   sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}'
>   sysctl vm.swapusage
>   echo ""
>   echo "### Servicios"
>   brew services list | grep -E "ollama|whisper" || echo "(ninguno)"
>   curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Ollama no responde"
>   echo ""
>   echo "### Toolchain"
>   python3 --version
>   uv --version 2>/dev/null || echo "uv no instalado"
>   huggingface-cli --version 2>/dev/null || echo "huggingface-cli no instalado"
>   ffmpeg -version 2>/dev/null | head -1 || echo "ffmpeg no instalado"
>   echo ""
>   echo "### Modelos Ollama"
>   ollama list 2>/dev/null || echo "Ollama no disponible"
> } > ejercicios/troubleshooting/diagnostico.md
> ```
>
> **Preguntas:**
> - ¿Tienes swap usado? Si sí, indica presión de memoria.
> - ¿Ollama responde en localhost:11434?
> - ¿Tienes compilados `llama-cli` y `main` (whisper.cpp)?
> - ¿Las herramientas (`uv`, `huggingface-cli`, `ffmpeg`) están instaladas?
>
> **Bonus:** reproduce y arregla deliberadamente uno de los errores de este anexo (p. ej. ejecuta `pip install` sin venv para ver PEP 668, o carga un modelo sin `--mmproj`). Documenta tu fix en `ejercicios/troubleshooting/caso_real.md`.

---

## Recursos

- **llama.cpp troubleshooting:** https://github.com/ggml-org/llama.cpp#troubleshooting
- **Ollama FAQ:** https://docs.ollama.com/faq
- **MLX issues:** https://github.com/ml-explore/mlx/issues
- **whisper.cpp issues:** https://github.com/ggml-org/whisper.cpp/issues
- **uv docs (troubleshooting):** https://docs.astral.sh/uv/reference/troubleshooting/
- **PEP 668 explicado:** https://peps.python.org/pep-0668/
- **HuggingFace CLI docs:** https://huggingface.co/docs/huggingface_hub/guides/cli
- **Guía de diagnóstico de memoria macOS:** `man memory_pressure`, `man top`
- **Foro de la comunidad del curso:** (enlazar a Discord/Telegram del curso)
- **Anexo relacionado:** [Glosario](B-Glosario.md) para términos técnicos

---

> [!tip] Siguiente paso
> Para definiciones de términos técnicos, consulta [Glosario](B-Glosario.md).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Modelos recomendados para Mac M2 24 GB](D-Modelos-para-Apple-Silicon-24GB.md) · [Índice](../README.md) · [Apéndice A - Fundamentos matemáticos →](F-Fundamentos-matematicos.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
