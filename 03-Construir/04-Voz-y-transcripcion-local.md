---
tags:
  - curso/ia-local
  - whisper
  - stt
  - audio
curso: IA-Local-de-Cero-a-Produccion
modulo: anexo-whisper-stt
estado: completo
---

# Voz y transcripción local con Whisper

<!-- CURSO_NAV_TOP -->
[← IA multimodal local: imagen, texto y audio](03-IA-multimodal-local.md) · [Índice](../README.md) · [04 - Fine-tuning en Mac →](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Windows y macOS
> whisper.cpp funciona en ambos sistemas y ofrece CPU, Metal, Vulkan, NVIDIA, AMD y OpenVINO según la compilación. `mlx-whisper` es solo para Apple Silicon.


> [!goals]
> **Objetivos de este anexo:**
> - Entender qué es STT (Speech-to-Text) y por qué Whisper cambió el panorama.
> - Conocer la arquitectura encoder-decoder de Whisper y su ventanas de 30 s.
> - Diferenciar entre transcripción batch y streaming.
> - Compilar y ejecutar whisper.cpp en macOS o Windows.
> - Instalar y usar mlx-whisper como alternativa nativa Apple Silicon.
> - Aplicar STT a casos reales: reuniones, subtítulos y dictado en vivo.

---


---

## Contexto pedagógico

### ¿Qué es STT?

**Speech-to-Text (STT)**, también llamado ASR (Automatic Speech Recognition), convierte audio hablado en texto. Es la tecnología que permite:

- Dictado de notas.
- Transcripción de reuniones y entrevistas.
- Subtítulos automáticos de vídeo.
- Comandos de voz en asistentes.
- Indexación y búsqueda de contenido de audio.

### Por qué Whisper es relevante

Whisper fue liberado por OpenAI en septiembre de 2022. Cambió el panorama por:

1. **Multilingüe por diseño:** soporta 99 idiomas sin necesidad de un modelo por idioma.
2. **Entrenado con 680 000 h de audio** supervisado débilmente — datos masivos y diversos.
3. **Robustez a ruido, acentos y audio no ideal** muy superior a modelos comerciales de la época.
4. **Abierto:** pesos del modelo publicados, lo que permitió portar a C++ (whisper.cpp), CoreML, MLX y ONNX.
5. **Detección de idioma automática y timestamps a nivel de palabra** (en variantes como whisper-timestamped).

### Diferencia entre streaming y batch

| Aspecto | Batch | Streaming |
|---|---|---|
| Procesa | Archivo completo de una vez | Chunks de audio en tiempo real |
| Latencia | Segundos a minutos | < 1 s idealmente |
| Calidad | Mejor (más contexto) | Ligeramente menor |
| Uso típico | Transcribir reunión grabada | Dictado en vivo, subtítulos en directo |
| Implementación | `whisper.cpp` / `mlx-whisper` estándar | Requiere buffer rodante + silero VAD |

### Modelos: de tiny a large-v3

| Modelo | Parámetros | RAM aprox. | Velocidad rel. | WER* (en) | Cuándo usar |
|---|---|---|---|---|---|
| tiny | 39 M | ~0,3 GB | 32× | ~12% | Dictado rápido, inglés limpio |
| base | 74 M | ~0,4 GB | 16× | ~10% | Español casual, buena calidad de audio |
| small | 244 M | ~0,9 GB | 6× | ~7% | Equilibrio en español |
| medium | 769 M | ~2,1 GB | 2× | ~5% | Buena calidad, multilingüe |
| large-v3 | 1550 M | ~3,9 GB | 1× | ~4% | Máxima calidad, si la velocidad es aceptable |

*WER = Word Error Rate, menor es mejor. Valores aproximados en inglés limpio.

> **Recomendación práctica:** empieza con `small`. Sube a `medium` o `large-v3` solo si la mejora de calidad compensa el tiempo. La tabla usa la memoria orientativa publicada por whisper.cpp; el backend y los buffers pueden añadir consumo.

---

## Profundización

### Arquitectura de Whisper

Whisper es un Transformer **encoder-decoder** (no es un decoder-only como los LLM típicos):

```
        Espectrograma Mel (80 bandas)
                    │
        ┌───────────▼───────────┐
        │  Audio Encoder         │
        │  (Transformer blocks)  │
        └───────────┬───────────┘
                    │
        Representaciones de audio
                    │
        ┌───────────▼───────────┐
        │  Text Decoder           │
        │  (Transformer blocks   │
        │   + cross-attention)   │
        └───────────┬───────────┘
                    │
                Texto + timestamps
```

1. **Frontend de audio:** el WAV/MP3 se re-muestrea a 16 kHz mono. Se calcula el **espectrograma Mel** de 80 bandas (representación tiempo-frecuencia).
2. **Encoder:** transforma el espectrograma en una secuencia de representaciones latentes que capturan fonemas, palabras y contexto acústico.
3. **Decoder:** genera tokens de texto usando **cross-attention** sobre la salida del encoder. Esto es lo que diferencia de un LLM puro: el decoder puede "mirar" al encoder.
4. **Tokens especiales:** Whisper inserta tokens como `<|startoftranscript|>`, `<|es|>` (idioma detectado), `<|0.00|>` (timestamps), `<|endoftranscript|>`.

### Ventanas de 30 segundos

Whisper procesa el audio en **chunks de 30 s**. Internamente:

- El audio total se divide en segmentos de 30 s (con solapamiento implícito vía padding).
- Cada chunk se codifica y decodifica independientemente.
- Para chunks sin voz completa, el decoder puede generar tokens de timestamp para alinear.

> [!warning]
> La ventana de 30 s es una **limitación estructural**: audios con pausas largas o silencios pueden generar alucinaciones (el modelo "inventa" texto para llenar la ventana). Solución: usar **VAD** (Voice Activity Detection) para partir el audio en segmentos con voz real.

### Detección de idioma

- En los primeros 30 s, el encoder produce un embedding.
- Un clasificador ligero sobre ese embedding predice el idioma.
- Si se fuerza un idioma con `-l es`, se omite la detección (más rápido y evita errores).
- Para español, fuerza siempre `-l es` salvo en audios multilingües.

### Timestamps

- Whisper genera timestamps a nivel de **segmento** (~ frase) por defecto.
- Para timestamps a nivel de **palabra**, usar:
  - `whisper-timestamped` (Python).
  - `whisper.cpp` con flag `-ml` (max_len) y postprocesado.
  - Modelos especiales como `large-v3-turbo` que incluyen mejor alineación.

---

## Comandos: whisper.cpp

### Compilación en macOS

```bash
cd ~/proyectos
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build
cmake --build build --config Release -j
./build/bin/whisper-cli --help
```

Metal se habilita en la ruta habitual de Apple Silicon. Si quieres una compilación específica, revisa las opciones actuales del repositorio.

### Compilación en Windows PowerShell

Necesitas Git, CMake y las herramientas C++ de Visual Studio:

```powershell
Set-Location "$HOME\proyectos"
git clone https://github.com/ggml-org/whisper.cpp
Set-Location whisper.cpp
cmake -B build
cmake --build build --config Release -j
Get-ChildItem .\build -Recurse -Filter whisper-cli.exe
```

Usa la ruta devuelta en el último paso. Con el generador de Visual Studio suele ser:

```powershell
.\build\bin\Release\whisper-cli.exe --help
```

### Descargar modelos

```bash
# Modelo multilingüe medium (buen equilibrio para español)
bash ./models/download-ggml-model.sh medium

# Modelo large-v3 (máxima calidad, ~3 GB)
bash ./models/download-ggml-model.sh large-v3

# Listar modelos descargados
ls -lh models/ggml-*.bin
```

En Windows nativo puedes descargar el modelo oficial directamente:

```powershell
New-Item -ItemType Directory -Force .\models
Invoke-WebRequest `
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" `
  -OutFile ".\models\ggml-small.bin"
```

### Transcribir archivo de audio real

```bash
# Asegurar audio en formato correcto (16 kHz, mono, WAV)
ffmpeg -i reunion.mp3 -ar 16000 -ac 1 -c:a pcm_s16le reunion_16k.wav

# Transcribir a texto
./build/bin/whisper-cli -m models/ggml-medium.bin -f reunion_16k.wav -l es -otxt

# Transcribir a SRT (subtítulos)
./build/bin/whisper-cli -m models/ggml-medium.bin -f reunion_16k.wav -l es -osrt

# Transcribir a JSON con timestamps
./build/bin/whisper-cli -m models/ggml-medium.bin -f reunion_16k.wav -l es -oj

# Con VAD activado (mejor para audios con silencios)
./build/bin/whisper-cli -m models/ggml-medium.bin -f reunion_16k.wav -l es -osrt \
  --prompt "Transcripción de reunión en español."
```

En Windows PowerShell:

```powershell
ffmpeg -i reunion.mp3 -ar 16000 -ac 1 -c:a pcm_s16le reunion_16k.wav
.\build\bin\Release\whisper-cli.exe `
  -m .\models\ggml-small.bin `
  -f .\reunion_16k.wav `
  -l es -otxt -osrt
```

### Verificar resultado

```bash
cat reunion_16k.txt
head -20 reunion_16k.srt
```

---

## Alternativa Apple Silicon: mlx-whisper

`mlx-whisper` corre Whisper nativamente sobre el framework MLX de Apple, optimizado para Apple Silicon. Suele ser más rápido que whisper.cpp en M-series para modelos grandes.

### Instalación

```bash
# Instalar como herramienta con uv
uv tool install mlx-whisper

# Verificar
mlx_whisper --help
```

### Transcribir

```bash
# Usa modelo large-v3 por defecto (se descarga solo la primera vez)
mlx_whisper reunion_16k.wav \
  --language es \
  --output-dir ./transcripciones \
  --output-format txt

# Forzar modelo específico
mlx_whisper reunion_16k.wav \
  --model mlx-community/Whisper-medium-mlx \
  --language es \
  --output-dir ./transcripciones \
  --output-format srt
```

### Benchmark rápido

```bash
# Comparar whisper.cpp vs mlx-whisper
time ./build/bin/whisper-cli -m models/ggml-medium.bin -f reunion_16k.wav -l es -otxt
time mlx_whisper reunion_16k.wav --model mlx-community/Whisper-medium-mlx \
  --language es --output-dir ./tmp
```

> [!tip]
> En M2 Pro/Max, `mlx-whisper` con large-v3 suele ser ~1.5–2× más rápido que whisper.cpp en el mismo modelo.

---

## Transcripción streaming (en tiempo real)

### Opción: streamer con whisper.cpp + ffmpeg

```bash
# Capturar 10 segundos del micrófono y transcribir
ffmpeg -f avfoundation -i ":0" -t 10 -ar 16000 -ac 1 -c:a pcm_s16le live.wav
./build/bin/whisper-cli -m models/ggml-medium.bin -f live.wav -l es -otxt
cat live.txt
```

### Opción: loop de dictado en vivo

```python
# dictado_live.py — loop de captura + transcripción con mlx-whisper
# Requisitos: uv pip install mlx-whisper sounddevice numpy
import subprocess
import tempfile
import time
from pathlib import Path

DURATION = 5  # segundos por chunk
MODEL = "mlx-community/Whisper-medium-mlx"

def grabar_y_transcribir():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    # Grabar con ffmpeg (macOS avfoundation)
    subprocess.run([
        "ffmpeg", "-y", "-f", "avfoundation", "-i", ":0",
        "-t", str(DURATION), "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", path
    ], capture_output=True)
    # Transcribir
    result = subprocess.run([
        "mlx_whisper", path, "--model", MODEL,
        "--language", "es", "--output-format", "txt",
        "--output-dir", str(Path(path).parent)
    ], capture_output=True, text=True)
    txt_path = Path(path).with_suffix(".txt")
    if txt_path.exists():
        print(txt_path.read_text(), end="", flush=True)
    Path(path).unlink(missing_ok=True)

if __name__ == "__main__":
    print("🎙️ Dictado en vivo (Ctrl+C para salir)\n")
    try:
        while True:
            grabar_y_transcribir()
    except KeyboardInterrupt:
        print("\n⛔ Detenido.")
```

Ejecutar:

```bash
uv run python dictado_live.py
```

> [!warning]
> Este es un loop simple con latencia de ~5 s (duración del chunk + transcripción). Para streaming verdadero (< 1 s de latencia) se necesita [whisper_streaming](https://github.com/ufal/whisper_streaming) o [WhisperLive](https://github.com/collabora/WhisperLive), con VAD y buffer rodante.

---

## Casos prácticos

### 1. Transcribir reuniones

```bash
# Grabar reunión (QuickTime o ffmpeg desde dispositivo de audio)
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -c:a pcm_s16le reunion.wav

# Transcribir con large-v3 (mejor calidad)
./build/bin/whisper-cli -m models/ggml-large-v3.bin -f reunion.wav -l es -osrt -oj

# Resumir con un LLM local (Ollama)
cat reunion.txt | ollama run qwen2.5:7b \
  "Resume esta transcripción en 5 puntos clave y lista las decisiones acordadas."
```

### 2. Subtítulos para vídeo

```bash
# Extraer audio de un vídeo
ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 -c:a pcm_s16le audio.wav

# Generar SRT
./build/bin/whisper-cli -m models/ggml-medium.bin -f audio.wav -l es -osrt

# El archivo audio.srt se puede incrustar en el vídeo:
ffmpeg -i video.mp4 -i audio.srt -c copy -c:s mov_text video_sub.mp4
```

### 3. Dictado en tiempo real (bocetos rápidos)

```bash
# Script mínimo de dictado (ver código Python arriba)
uv run python dictado_live.py
```

---

## Ejercicio práctico

> [!exercise]
> **Transcribe y resume una reunión de 5 minutos**
>
> 1. Graba 5 minutos de audio (puede ser un podcast en inglés o tu propia voz en español):
>    ```bash
>    ffmpeg -f avfoundation -i ":0" -t 300 -ar 16000 -ac 1 -c:a pcm_s16le mi_reunion.wav
>    ```
> 2. Transcribe con whisper.cpp usando **small**:
>    ```bash
>    cd ~/proyectos/whisper.cpp
>    ./build/bin/whisper-cli -m models/ggml-small.bin -f mi_reunion.wav -l es -osrt -otxt
>    ```
> 3. Repite con **medium** y compara:
>    ```bash
>    ./build/bin/whisper-cli -m models/ggml-medium.bin -f mi_reunion.wav -l es -osrt -otxt
>    ```
> 4. Genera un resumen con un LLM local:
>    ```bash
>    cat mi_reunion.txt | ollama run qwen2.5:7b \
>      "Resume esta transcripción en español. Lista: (1) temas tratados, (2) decisiones, (3) tareas pendientes."
>    ```
> 5. **Entregable:** guarda en `ejercicios/stt/`:
>    - `mi_reunion_small.srt`
>    - `mi_reunion_medium.srt`
>    - `resumen.md`
>
> **Preguntas:**
> - ¿Qué diferencia de calidad notas entre small y medium?
> - ¿Cuánto tarda cada uno (usa `time` antes del comando)?
> - ¿El LLM resumió correctamente o hallucinó?
>
> **Bonus:** transcribe el mismo audio con `mlx-whisper` y compara tiempo.

---

## Recursos

- **whisper.cpp:** https://github.com/ggml-org/whisper.cpp
- **mlx-whisper:** https://github.com/ml-explore/mlx-examples/tree/main/whisper
- **Whisper original (OpenAI):** https://github.com/openai/whisper
- **Whisper paper:** *Robust Speech Recognition via Large-Scale Weak Supervision* (Radford et al., 2022) — https://cdn.openai.com/papers/whisper.pdf
- **Modelos GGUF pre-cuantizados:** https://huggingface.co/ggerganov/whisper.cpp
- **Modelos MLX:** https://huggingface.co/mlx-community (buscar "Whisper")
- **whisper_streaming (low latency):** https://github.com/ufal/whisper_streaming
- **whisper-timestamped:** https://github.com/linto-ai/whisper-timestamped
- **Silero VAD (para segmentar audio):** https://github.com/snakers4/silero-vad
- **Benchmarks de whisper.cpp:** https://github.com/ggml-org/whisper.cpp#benchmarks

---

> [!tip] Siguiente paso
> Si te encuentras errores durante la compilación o ejecución, consulta [Troubleshooting](../07-Anexos/E-Troubleshooting-local.md).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← IA multimodal local: imagen, texto y audio](03-IA-multimodal-local.md) · [Índice](../README.md) · [04 - Fine-tuning en Mac →](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
