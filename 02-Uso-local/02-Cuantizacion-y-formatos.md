---
tags:
  - curso/ia-local
  - cuantizacion
  - gguf
  - mlx
curso: IA-Local-de-Cero-a-Produccion
modulo: 03
estado: completo
---

# 03 - Cuantización y reducción de memoria

<!-- CURSO_NAV_TOP -->
[← 02 - Inferencia local](01-Inferencia-con-Ollama-llama.cpp-y-MLX.md) · [Índice](../README.md) · [07 - RAG local: pipelines con contexto →](../03-Construir/01-RAG-local.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Windows y macOS
> Las partes de Ollama, llama.cpp y Python sirven en ambos sistemas. MLX y Metal son exclusivos de Apple Silicon; en Windows usa CUDA, Vulkan o CPU según tu equipo. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!goals] Objetivos de aprendizaje
> - Entender qué cambia al pasar de FP16 a INT8/INT4.
> - Elegir niveles GGUF adecuados para 24 GB: Q4_K_M, Q5_K_M, Q6 y Q8.
> - Cuantizar modelos con llama.cpp y MLX.
> - Medir calidad, RAM y velocidad en tu propia máquina.



## Contexto pedagógico: cuantizar es comprimir con criterio

Cuantizar no es simplemente “hacer el archivo más pequeño”. Es representar los pesos con menos precisión numérica. El modelo sigue teniendo la misma arquitectura, pero algunos números se guardan con menos bits.

La intuición:

```text
FP16: muchos matices numéricos, mucha memoria
INT8: menos matices, menos memoria
INT4: muchos menos matices, mucha menos memoria
```

El truco es que las redes neuronales suelen tolerar bastante ruido en los pesos. No todos los pesos necesitan precisión FP16 para producir una respuesta útil. La cuantización explota esa tolerancia.

## Qué pasa realmente al cuantizar

Un peso FP16 puede tomar muchísimos valores. En 4 bits solo tienes 16 niveles posibles por bloque o grupo, más escalas auxiliares. Los métodos modernos no redondean todo de forma ingenua: agrupan pesos, calculan escalas y preservan mejor las partes sensibles.

Por eso dos cuantizaciones “4-bit” pueden comportarse de forma distinta. Q4_K_M no es lo mismo que una cuantización INT4 simple. La calidad depende del esquema, del modelo y de la tarea.

## La tensión central de este módulo

```text
menos bits
  → menos memoria y a veces más velocidad
  → más error numérico
  → posible pérdida de calidad
```

Tu objetivo no es usar siempre la cuantización más pequeña. Es encontrar el punto donde la pérdida no importa para tu caso y el ahorro sí importa para tu Mac.

## Por qué 24 GB cambia las prioridades

En una GPU grande puedes permitirte FP16 o cuantizaciones suaves. En un Mac M2 de 24 GB, el límite es práctico: si el modelo no cabe con margen, la experiencia se degrada. Por eso [03-Cuantizacion](02-Cuantizacion-y-formatos.md) es el módulo central para IA local.

## 1. Por qué cuantizar

Los pesos de un modelo suelen entrenarse en FP16/BF16. Eso ocupa unos 2 bytes por parámetro.

```text
7B FP16 ≈ 14 GB
7B Q4 ≈ 4-5 GB
```

En un Mac M2 con 24 GB, cuantizar no es opcional si quieres usar modelos 7B-14B con margen para contexto, sistema y aplicaciones. Para el cálculo detallado del KV cache, ver [Memoria-KV-Cache-Apple-Silicon](../01-Fundamentos/03-Memoria-contexto-y-KV-cache-en-Apple-Silicon.md).

## 2. Qué se pierde y qué se gana

Ganas:

- menor RAM;
- menor presión de memoria;
- más modelos viables;
- a veces más velocidad por menor ancho de banda.

Pierdes:

- algo de precisión;
- más degradación en tareas difíciles;
- posibles errores raros en razonamiento, matemáticas o código;
- peor comportamiento si cuantizas demasiado.

La métrica técnica habitual es perplexity, pero para uso real necesitas también evals de tus tareas. Volvemos a esto en [06-Proyecto-Final](../06-Proyectos/01-Asistente-local-completo.md).

## 3. Niveles GGUF prácticos

Guía rápida:

| Nivel | Uso recomendado | Comentario |
|---|---|---|
| Q4_K_M | predeterminado para 24 GB | buen equilibrio calidad/RAM |
| Q5_K_M | si quieres más calidad | algo más pesado |
| Q6_K | calidad alta | viable en 7B, más justo en 14B |
| Q8_0 | casi FP16 práctico | pesado; úsalo para modelos pequeños |
| F16 | referencia o conversión | no ideal para inferencia en 24 GB |

Para 24 GB:

- 7B: Q4_K_M, Q5_K_M o Q6_K.
- 14B: Q4_K_M o Q5_K_M con contexto moderado.
- 30B+: no recomendable localmente.

## 4. AWQ, GPTQ, GGUF y MLX

- **GGUF**: ideal con llama.cpp/Ollama/LM Studio.
- **AWQ/GPTQ**: comunes en GPU NVIDIA; no son la ruta principal en Mac.
- **MLX 4-bit/8-bit**: práctico para inferencia y LoRA en Apple Silicon.

En Mac M2, prioriza GGUF o MLX salvo que tengas una razón concreta.

## 5. Preparar llama.cpp para cuantizar

Si compilaste en 00-Setup, ya deberías tener binarios. Verifica:

```bash
cd ~/ia-local/llama.cpp
./build/bin/llama-quantize --help | head
./build/bin/llama-cli --help | head
```

Necesitas primero convertir desde Hugging Face a GGUF F16/BF16:

Usa siempre el entorno de `uv` para no tocar el Python del sistema:

```bash
cd ~/ia-local/curso
source .venv/bin/activate
uv pip install -r ~/ia-local/llama.cpp/requirements.txt

# Verifica la ruta del script: en algunas versiones puede estar en examples/ o tener otro nombre
ls ~/ia-local/llama.cpp/convert_hf_to_gguf.py 2>/dev/null || ls ~/ia-local/llama.cpp/examples/convert_hf_to_gguf.py 2>/dev/null

python ~/ia-local/llama.cpp/convert_hf_to_gguf.py ~/ia-local/models/modelo-hf \
  --outfile ~/ia-local/models/modelo-f16.gguf \
  --outtype f16
```

## 6. Cuantizar a Q4_K_M con llama.cpp

```bash
cd ~/ia-local/llama.cpp
./build/bin/llama-quantize \
  ~/ia-local/models/modelo-f16.gguf \
  ~/ia-local/models/modelo-Q4_K_M.gguf \
  Q4_K_M
```

Prueba el resultado:

```bash
./build/bin/llama-cli \
  -m ~/ia-local/models/modelo-Q4_K_M.gguf \
  -p "Explica la cuantización en dos frases." \
  -n 120 \
  -ngl 99 \
  -c 4096
```

Si el modelo repite o responde peor que el original, prueba Q5_K_M.

## 7. Cuantización con MLX

Convertir un modelo Hugging Face a MLX 4-bit:

```bash
cd ~/ia-local/curso
source .venv/bin/activate
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-3B-Instruct \
  --mlx-path ~/ia-local/models/qwen2.5-3b-mlx-4bit \
  -q
```

Genera:

```bash
mlx_lm.generate \
  --model ~/ia-local/models/qwen2.5-3b-mlx-4bit \
  --prompt "Define cuantización en una frase." \
  --max-tokens 80
```

Para 7B, también es viable en 24 GB. Para 14B, vigila memoria y reduce batch/contexto si hace falta.

## 8. Cuantización de KV cache

La KV cache crece con el contexto. Algunas herramientas permiten cuantizarla o usar tipos más pequeños.

En llama.cpp, revisa opciones actuales:

```bash
cd ~/ia-local/llama.cpp
./build/bin/llama-cli --help | grep -i "kv\|cache\|type"
```

Si existen flags como `--cache-type-k` o `--cache-type-v`, puedes probar:

```bash
./build/bin/llama-cli \
  -m ~/ia-local/models/modelo-Q4_K_M.gguf \
  -p "Resume este concepto." \
  -n 120 \
  -c 8192 \
  -ngl 99 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

No lo uses a ciegas: compara calidad y estabilidad. En modelos pequeños, el ahorro puede no compensar.

## 9. Medir calidad/RAM/velocidad

Crea una tabla propia, porque cada Mac y cada versión cambian.

```bash
/usr/bin/time -l ~/ia-local/llama.cpp/build/bin/llama-cli \
  -m ~/ia-local/models/modelo-Q4_K_M.gguf \
  -p "Escribe 5 bullets sobre Apple Silicon." \
  -n 200 \
  -ngl 99 \
  -c 4096
```

En macOS, `/usr/bin/time -l` muestra memoria máxima residente.

Plantilla:

| Modelo | Quant | Contexto | Tokens/s | Memoria máx. | Calidad subjetiva |
|---|---:|---:|---:|---:|---|
| Qwen2.5 7B | Q4_K_M | 4096 | medir | medir | buena |

## Profundización: error de cuantización y sensibilidad por capa

Cuantizar introduce error numérico. El punto importante es que ese error no afecta igual a todo. Algunas capas, matrices o tipos de pesos son más sensibles que otros. Los métodos modernos intentan conservar mejor las partes críticas y comprimir más las tolerantes.

Por eso existen esquemas con nombres aparentemente crípticos como Q4_K_M. No son simples “4 bits”. Incluyen decisiones sobre bloques, escalas, mezcla de precisión y tratamiento de matrices.

### Perplexity y calidad práctica

La perplexity mide lo bien que el modelo predice texto en un conjunto de evaluación. Si al cuantizar sube mucho, suele indicar pérdida. Pero una pequeña diferencia de perplexity no siempre se nota en tu aplicación.

Para tareas reales importa medir también:

- si sigue instrucciones;
- si mantiene formato;
- si razona suficiente para tu caso;
- si alucina más;
- si gana velocidad o estabilidad.

La pregunta no es “¿cuál es la cuantización objetivamente mejor?”, sino:

```text
¿Cuál es la mínima precisión que conserva calidad suficiente para mi tarea?
```

### Por qué Q4_K_M suele ser el punto de partida

Q4_K_M suele funcionar bien porque reduce mucho memoria sin destruir demasiada calidad en modelos 7B-14B. No es siempre óptimo, pero es un buen baseline.

- Si Q4_K_M falla en calidad, prueba Q5_K_M.
- Si Q5_K_M no cabe o va lento, vuelve a Q4.
- Si el modelo es pequeño y te sobra memoria, Q8 puede ser útil como referencia.

### Cuantización de pesos vs cuantización de KV cache

Hay dos consumos distintos:

```text
pesos del modelo → dependen del tamaño y cuantización del modelo
KV cache → depende de contexto, capas, dimensión y tipo de cache
```

Cuantizar pesos reduce el coste fijo de cargar el modelo. Cuantizar KV cache reduce el coste variable de mantener contexto largo. Son problemas relacionados, pero no iguales.

### Efectos cualitativos que debes observar

La mala cuantización no siempre produce errores obvios. A veces aparece como:

- respuestas más genéricas;
- peor seguimiento de instrucciones largas;
- más repeticiones;
- peor razonamiento matemático;
- degradación en código;
- mayor sensibilidad al prompt.

Por eso conviene mantener un pequeño set de prompts de regresión y repetirlos cuando cambies quant.

## Ejercicio práctico

1. Descarga un modelo GGUF en Q4_K_M y Q5_K_M.
2. Ejecuta el mismo prompt en ambos.
3. Mide memoria con `/usr/bin/time -l` o Activity Monitor.
4. Decide cuál usarías por defecto.
5. Si tienes tiempo, convierte un modelo 3B a MLX 4-bit.

Prompt de comparación:

```text
Explica cómo elegir entre Q4_K_M y Q5_K_M para un Mac con 24 GB, con una recomendación final.
```

## Recursos

- llama.cpp quantization: https://github.com/ggml-org/llama.cpp
- TheBloke GGUF notes históricas: https://huggingface.co/TheBloke
- Hugging Face quantization overview: https://huggingface.co/docs/transformers/quantization
- MLX LM quantization: https://github.com/ml-explore/mlx-lm

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← 02 - Inferencia local](01-Inferencia-con-Ollama-llama.cpp-y-MLX.md) · [Índice](../README.md) · [07 - RAG local: pipelines con contexto →](../03-Construir/01-RAG-local.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
