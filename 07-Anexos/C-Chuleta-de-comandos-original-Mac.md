---
tags:
  - curso/ia-local
  - chuleta
  - comandos
curso: IA-Local-de-Cero-a-Produccion
modulo: anexo-chuleta
estado: completo
---

# Chuleta de comandos

<!-- CURSO_NAV_TOP -->
[← Glosario](B-Glosario.md) · [Índice](../README.md) · [Modelos recomendados para Mac M2 24 GB →](D-Modelos-para-Apple-Silicon-24GB.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Ruta Apple Silicon
> Este capítulo nació para Mac con Apple Silicon. Si usas Windows, quédate con los conceptos y sigue la alternativa indicada en [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!goals] Objetivos de aprendizaje
> - Tener a mano comandos ejecutables para Ollama, llama.cpp y MLX.
> - Reducir fricción al repetir pruebas.
> - Usar esta nota como referencia rápida durante el curso.


## Setup rápido

```bash
xcode-select --install
brew update
brew install git git-lfs cmake ninja pkg-config python@3.11 uv wget curl jq htop
git lfs install
```

```bash
mkdir -p ~/ia-local/curso
cd ~/ia-local/curso
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install mlx mlx-lm huggingface_hub datasets transformers sentencepiece protobuf
```

Relacionado: 00-Setup.

## Ollama

Arrancar:

```bash
open -a Ollama
```

Modelos:

```bash
ollama list
ollama pull qwen2.5:7b-instruct
ollama show qwen2.5:7b-instruct
ollama run qwen2.5:7b-instruct "Hola, responde en una frase."
ollama rm qwen2.5:7b-instruct
```

API:

```bash
curl http://localhost:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:7b-instruct","prompt":"Define GGUF.","stream":false}' | jq
```

Modelfile:

```bash
cat > Modelfile <<'EOF'
FROM qwen2.5:7b-instruct
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
SYSTEM "Eres un asistente técnico directo."
EOF

ollama create qwen-tecnico -f Modelfile
ollama run qwen-tecnico "Dame una recomendación de cuantización."
```

Relacionado: [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md).

## llama.cpp

Compilar con Metal:

```bash
cd ~/ia-local
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Ejecutar GGUF:

```bash
cd ~/ia-local/llama.cpp
./build/bin/llama-cli \
  -m ~/ia-local/models/modelo-Q4_K_M.gguf \
  -p "Explica qué es la cuantización." \
  -n 200 \
  -ngl 99 \
  -fa \
  -c 4096
```

Servidor local:

```bash
./build/bin/llama-server \
  -m ~/ia-local/models/modelo-Q4_K_M.gguf \
  -ngl 99 \
  -fa \
  -c 4096 \
  --host 127.0.0.1 \
  --port 8080
```

Llamada OpenAI-compatible:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hola"}],"temperature":0.2}' | jq
```

Convertir y cuantizar:

```bash
cd ~/ia-local/llama.cpp
source ~/ia-local/curso/.venv/bin/activate
python convert_hf_to_gguf.py ~/ia-local/models/modelo-hf \
  --outfile ~/ia-local/models/modelo-f16.gguf \
  --outtype f16

./build/bin/llama-quantize \
  ~/ia-local/models/modelo-f16.gguf \
  ~/ia-local/models/modelo-Q4_K_M.gguf \
  Q4_K_M
```

Relacionado: [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

## MLX / mlx-lm

Generar:

```bash
cd ~/ia-local/curso
source .venv/bin/activate
mlx_lm.generate \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --prompt "Explica MLX en una frase." \
  --max-tokens 120 \
  --temp 0.2
```

Servidor:

```bash
mlx_lm.server \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8081
```

Convertir a MLX 4-bit:

```bash
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-3B-Instruct \
  --mlx-path ~/ia-local/models/qwen2.5-3b-mlx-4bit \
  -q
```

LoRA:

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --train \
  --data ~/ia-local/fine-tuning/data \
  --iters 300 \
  --batch-size 1 \
  --lora-layers 16 \
  --learning-rate 1e-5 \
  --adapter-path ~/ia-local/fine-tuning/adapters/demo-lora
```

Fusión:

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --adapter-path ~/ia-local/fine-tuning/adapters/demo-lora \
  --save-path ~/ia-local/fine-tuning/fused/demo
```

Relacionado: [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

## Medición

```bash
/usr/bin/time -l ~/ia-local/llama.cpp/build/bin/llama-cli \
  -m ~/ia-local/models/modelo-Q4_K_M.gguf \
  -p "Dame 5 bullets sobre Apple Silicon." \
  -n 200 \
  -ngl 99 \
  -c 4096
```

```bash
memory_pressure
vm_stat
top -l 1 | head -n 20
```

Con `asitop`:

```bash
source ~/ia-local/curso/.venv/bin/activate
uv pip install asitop
sudo asitop
```

## Ejercicio práctico

Copia esta nota, elimina los comandos que no uses y deja tu propia chuleta mínima con los modelos y rutas reales de tu Mac.

## Recursos

- 00-Setup
- [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md)
- [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md)
- [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md)
- llama.cpp: https://github.com/ggml-org/llama.cpp
- MLX LM: https://github.com/ml-explore/mlx-lm
- Ollama: https://ollama.com/

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Glosario](B-Glosario.md) · [Índice](../README.md) · [Modelos recomendados para Mac M2 24 GB →](D-Modelos-para-Apple-Silicon-24GB.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
