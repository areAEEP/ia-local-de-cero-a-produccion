---
tags:
  - curso/ia-local
  - multimodal
  - vision
  - whisper
curso: IA-Local-de-Cero-a-Produccion
modulo: anexo-multimodal
estado: completo
---

# IA multimodal local: imagen, texto y audio

<!-- CURSO_NAV_TOP -->
[← 08 - Agentes locales: LLM + herramientas](02-Agentes-locales-y-MCP.md) · [Índice](../README.md) · [Voz y transcripción local con Whisper →](04-Voz-y-transcripcion-local.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Linux, Windows y macOS
> Ollama y el ejemplo Python funcionan en los tres sistemas. Los apartados MLX/Metal son la ruta Apple; en Linux o Windows usa el backend compatible con tu GPU. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!goals]
> **Objetivos de este anexo:**
> - Entender qué es un modelo multimodal y por qué difiere de un LLM texto-only.
> - Conocer la arquitectura interna de los modelos vision-language (vision encoder + projector + LLM).
> - Identificar modelos multimodales viables según la memoria del equipo.
> - Ejecutar un VLM local con Ollama y enviar imágenes por API.
> - Instalar y usar Whisper (whisper.cpp / mlx-whisper) para transcripción de audio local.
> - Combinar visión + audio en un pipeline local funcional.

---


---

## Contexto pedagógico

### ¿Qué es multimodal?

Un modelo **multimodal** puede procesar más de un tipo de dato como entrada: texto, imágenes, audio, vídeo. En el contexto de IA local, los dos modos más prácticos son:

1. **Visión (image-understanding):** el modelo "ve" una imagen y responde preguntas sobre ella.
2. **Audio → texto (STT):** transcribir voz a texto, habitualmente con Whisper.

### Por qué importa

Un LLM texto-only (p. ej. Llama 3.1 8B base) solo opera sobre tokens de texto. No puede interpretar un diagrama, leer una factura escaneada, describir una captura de pantalla ni transcribir una reunión. La capacidad multimodal amplía el rango de tareas a:

- OCR y extracción de datos de documentos.
- Descripción de imágenes para accesibilidad.
- Análisis de capturas de UI para debugging.
- Razonamiento sobre diagramas técnicos.
- Transcripción y resumen de reuniones.

### Diferencia entre modelo texto-only y modelo que entiende imágenes

| Aspecto | LLM texto-only | VLM (Vision-Language Model) |
|---|---|---|
| Entradas | Tokens de texto | Tokens de texto + tokens de imagen |
| Componentes | Solo decoder transformer | Vision encoder + projector + LLM |
| Ejemplo local | Llama 3.1 8B, Qwen2.5 7B | Qwen2-VL, LLaVA, MiniCPM-V |
| RAM típica (Q4) | 4–6 GB | 6–10 GB (2B) / 12–18 GB (7B) |
| Caso de uso | Chat, código, razonamiento | Lo anterior + imágenes |

---

## Profundización

### Arquitectura de un modelo vision-language

Un VLM moderno tiene tres bloques:

```
┌─────────────┐     ┌────────────┐     ┌─────────────┐
│  Vision      │────▶│  Projector  │────▶│   LLM        │
│  Encoder     │     │  (MLP)      │     │   (decoder)  │
│  (ViT/CLIP)  │     └────────────┘     └─────────────┘
└─────────────┘
   Imagen          Tokens visuales         Tokens de texto
                                          + tokens visuales
```

1. **Vision Encoder** — Habitualmente un ViT (Vision Transformer) preentrenado tipo CLIP. Convierte la imagen en una secuencia de representaciones latentes.
2. **Projector / Adapter** — Una capa (a menudo un MLP) que proyecta las representaciones visuales al espacio de embeddings del LLM. Es el "puente" entre ambos mundos. Su calidad determina cuánta información visual sobrevive al paso al LLM.
3. **LLM** — El decoder de lenguaje (p. ej. Qwen2) que recibe como entrada una secuencia que mezcla tokens de texto y tokens de imagen, y genera la respuesta textual.

### Cómo se tokenizan las imágenes (patches)

El vision encoder **no** recibe píxeles sueltos. La imagen se divide en una cuadrícula de **patches** (típicamente 14×14 o 16×16 píxeles). Cada patch se aplana y se proyecta a un embedding, igual que un token de texto.

Ejemplo simplificado:

- Imagen de entrada: 448 × 448 px.
- Patch size: 14 × 14 → 32 × 32 = **1024 patches**.
- Cada patch → 1 embedding de dimensión *d* (p. ej. 1152 en ViT-L).
- Tras el projector, esos 1024 vectores entran al LLM como "tokens visuales".

> [!note]
> Los VLM dinámicos (Qwen2-VL) ajustan la resolución de cada imagen al contenido: imágenes grandes generan más tokens visuales y consumen más contexto y más RAM. Una sola imagen de alta resolución puede ocupar **4 000–8 000 tokens** del contexto.

### Limitaciones

- **Alucinación visual:** el modelo inventa objetos no presentes. Más común que en texto puro.
- **Texto fino (OCR):** los VLM pequeños (2B) fallan en texto denso o pequeño. Mejor OCR dedicado (Tesseract, PaddleOCR) para extracción precisa.
- **Conteo y relaciones espaciales:** "¿cuántas personas?" o "¿qué hay a la izquierda de...?" son tareas donde los VLM pequeños fallan con frecuencia.
- **Latencia:** procesar imágenes es 3–10× más lento que texto equivalente.
- **Ventana de contexto:** los tokens visuales consumen contexto rápidamente. 8 imágenes pueden saturar un contexto de 8 K.
- **Facturación de RAM:** el vision encoder añade 0.5–2 GB adicionales al peso del modelo.

---

## Modelos viables según tu equipo

| Modelo | Parámetros | RAM Q4 aprox. | Notas |
|---|---|---|---|
| Qwen2-VL 2B | 2 B | ~3–4 GB | Rápido, buen baseline. Limitado en OCR. |
| Qwen2-VL 7B | 7 B | ~6–8 GB | Mejor equilibrio calidad/velocidad. |
| Qwen2.5-VL 7B | 7 B | ~6–8 GB | Versión mejorada, mejor OCR y razonamiento espacial. |
| LLaVA 1.6 7B | 7 B | ~6–8 GB | Soporte amplio en llama.cpp/Ollama. |
| LLaVA 1.6 13B | 13 B | ~10–12 GB | Cabe pero deja poco margen. |
| MiniCPM-V 2.6 | 8 B | ~5–7 GB | Excelente OCR y multi-imagen. Muy optimizado. |
| MiniCPM-V 4.0 | 8 B | ~5–7 GB | Última versión, muy capaz. |

> **Recomendación para M2 24 GB:** Qwen2.5-VL 7B Q4 o MiniCPM-V para OCR denso. Qwen2-VL 2B si necesitas velocidad máxima.

Para una primera prueba actual y común a Windows/macOS, busca un tag pequeño de **Qwen 3.5** o **Gemma** con capacidad `vision` en la [biblioteca oficial de Ollama](https://ollama.com/search). Los tags cambian con el tiempo: verifica la ficha antes de copiar el nombre.

---

## Comandos: visión con Ollama

### Instalar/ejecutar vía Ollama

```bash
# Descargar un VLM pequeño. Comprueba primero que el tag existe.
ollama pull qwen3.5:4b
ollama list

# Ejecutar interactivo
ollama run qwen3.5:4b "Describe esta imagen" ./foto.jpg
```

### Enviar imagen vía API HTTP de Ollama

```bash
# Codificar imagen a base64
IMG_B64=$(base64 -i ./captura.png)

curl -s http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:4b",
  "messages": [
    {
      "role": "user",
      "content": "¿Qué muestra esta captura de pantalla? Lista los elementos UI visibles.",
      "images": ["'"$IMG_B64"'"]
    }
  ],
  "stream": false
}' | python3 -m json.tool
```

### Código Python de ejemplo

```python
# multivision.py — Enviar imagen a Qwen2-VL vía Ollama
# Requisitos: pip install requests pillow
import base64
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"

def image_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

def ask_image(image_path: str, question: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": question,
            "images": [image_to_b64(image_path)],
        }],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]

if __name__ == "__main__":
    imagen = "./captura.png"   # ← cambia por tu imagen
    pregunta = "Describe la imagen en español. ¿Qué texto aparece?"
    print(ask_image(imagen, pregunta))
```

Ejecutar en Linux/macOS o Windows PowerShell:

```text
uv run python multivision.py
```

### Alternativa avanzada: llama.cpp

El soporte multimodal y los nombres de los binarios evolucionan deprisa. Empieza por la [documentación actual de llama.cpp](https://github.com/ggml-org/llama.cpp) y descarga juntos el GGUF y su projector `mmproj`. La idea es la misma en los tres sistemas; cambia la ruta al ejecutable y el backend:

```bash
# Clonar y compilar con Metal
cd ~/proyectos
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j

# Después de descargar el GGUF y su mmproj según la ficha del modelo:
./llama-cli \
  -m ./modelos/modelo-vision.gguf \
  --mmproj ./modelos/mmproj-modelo-vision.gguf \
  --image ./foto.jpg \
  -p "Describe esta imagen en español." \
  -n 512
```

> [!warning]
> Muchos VLM en llama.cpp necesitan **dos** archivos: el modelo principal y el vision projector (`mmproj-*.gguf`). Sin el projector, el modelo de lenguaje no puede interpretar la imagen. Sigue la ficha del GGUF elegido, porque algunos modelos modernos se empaquetan de otra forma.

---

## Whisper local (introducción)

Para el audio → texto, Whisper es el estándar abierto. En este anexo lo introducimos; el detalle completo está en [Whisper-STT-Local](04-Voz-y-transcripcion-local.md).

### Opción A: whisper.cpp

```bash
cd ~/proyectos
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build
cmake --build build --config Release -j

# Descargar el modelo multilingüe base
sh ./models/download-ggml-model.sh base

# Transcribir
./build/bin/whisper-cli -m models/ggml-base.bin -f audio.wav -l es -osrt
```

En Windows puedes compilar el mismo repositorio con CMake; el ejecutable suele quedar en `build\bin\Release\whisper-cli.exe`. El capítulo siguiente contiene el recorrido completo.

### Opción B: mlx-whisper (nativo Apple Silicon)

```bash
# Instalar con uv
uv tool install mlx-whisper

# Transcribir archivo (descarga modelo automáticamente)
mlx_whisper --model mlx-community/Whisper-large-v3-mlx \
  audio.mp3 --language es --output-dir ./transcripciones
```

---

## Ejercicio práctico

> [!exercise]
> **Pipeline multimodal: imagen + audio → resumen**
>
> 1. Toma una captura de pantalla de tu escritorio (`Cmd+Shift+3`).
> 2. Graba 30 segundos de voz explicando qué estás viendo (puedes usar `QuickTime` o `say` inverso).
>    ```bash
>    # Grabar audio del micrófono (10 s) — requiere soporte de micrófono
>    ffmpeg -f avfoundation -i ":0" -t 10 -ar 16000 audio.wav
>    ```
> 3. Transcribe el audio con whisper.cpp:
>    ```bash
>    cd ~/proyectos/whisper.cpp
>    ./build/bin/whisper-cli -m models/ggml-base.bin -f audio.wav -l es -otxt
>    ```
> 4. Envía la captura de pantalla a Qwen2-VL con la transcripción como contexto:
>    ```bash
>    IMG_B64=$(base64 -i "Captura de pantalla.png")
>    TRANSCRIPCION=$(cat audio.txt)
>    curl -s http://localhost:11434/api/chat -d '{
>      "model": "qwen3.5:4b",
>      "messages": [{
>        "role": "user",
>        "content": "Esta imagen es una captura de mi pantalla. El usuario dijo: «'"$TRANSCRIPCION"'». Resume en 3 puntos qué muestra y qué acción recomiendas.",
>        "images": ["'"$IMG_B64"'"]
>      }],
>      "stream": false
>    }' | python3 -m json.tool
>    ```
> 5. **Entregable:** guarda el JSON de salida en `ejercicios/multimodal_resultado.json` y responde:
>    - ¿El modelo identificó correctamente los elementos de la captura?
>    - ¿Hubo alucinación visual?
>    - ¿Cuánto tardó (latencia) en responder?
>
> **Bonus:** repite con Qwen2-VL 2B y compara calidad y velocidad.

---

## Recursos

- **Qwen2-VL repo:** https://github.com/QwenLM/Qwen2-VL
- **Modelos multimodales en Ollama:** https://ollama.com/search
- **llama.cpp multimodal:** https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd
- **MiniCPM-V:** https://github.com/OpenBMB/MiniCPM-V
- **LLaVA:** https://github.com/haotian-liu/LLaVA
- **whisper.cpp:** https://github.com/ggml-org/whisper.cpp
- **mlx-whisper:** https://github.com/ml-explore/mlx-examples/tree/main/whisper
- **OpenAI Whisper paper:** *Robust Speech Recognition via Large-Scale Weak Supervision* (Radford et al., 2022)
- **Vision Transformer paper:** *An Image is Worth 16x16 Words* (Dosovitskiy et al., 2020)
- **CLIP paper:** *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021)

---

> [!tip] Siguiente paso
> Profundiza en Whisper en el anexo dedicado: [Whisper-STT-Local](04-Voz-y-transcripcion-local.md).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← 08 - Agentes locales: LLM + herramientas](02-Agentes-locales-y-MCP.md) · [Índice](../README.md) · [Voz y transcripción local con Whisper →](04-Voz-y-transcripcion-local.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
