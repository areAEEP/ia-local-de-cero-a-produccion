---
tags:
  - curso/ia-local
  - proyecto-final
  - lora
  - evaluacion
  - ollama
curso: IA-Local-de-Cero-a-Produccion
modulo: 06
estado: completo
---

# 06 - Proyecto final

<!-- CURSO_NAV_TOP -->
[← Evaluación y monitorización de calidad](../05-LLMOps/12-Evaluacion-y-calidad-en-produccion.md) · [Índice](../README.md) · [P1 - Proyecto - Motor de inferencia desde cero →](02-Motor-de-inferencia-desde-cero.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Windows y macOS
> Las partes de Ollama, llama.cpp y Python sirven en ambos sistemas. MLX y Metal son exclusivos de Apple Silicon; en Windows usa CUDA, Vulkan o CPU según tu equipo. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!goals] Objetivos de aprendizaje
> - Construir un pipeline completo con un caso real.
> - Elegir modelo base, crear dataset, entrenar LoRA, fusionar, cuantizar y servir.
> - Conectar el modelo a una API local o agente.
> - Evaluar contra el modelo base con criterios claros.



## Contexto pedagógico: pasar de demo a sistema

El proyecto final existe para evitar el error más común: confundir una demo que responde bonito con un sistema útil.

Un sistema local completo tiene varias piezas:

```text
modelo base
  → datos de adaptación
  → entrenamiento o configuración
  → cuantización/despliegue
  → interfaz de uso
  → evaluación
  → decisión de si merece la pena
```

La parte importante no es que el modelo responda. Eso ya lo hacía antes. La parte importante es demostrar que, para tu caso, el pipeline mejora algo medible sin romper demasiado lo demás.

## Qué significa “mejorar”

Mejorar no significa que una respuesta aislada te guste más. Significa que, en un conjunto de casos representativos, el sistema:

- acierta más;
- sigue mejor el formato;
- pide menos aclaraciones innecesarias;
- alucina menos;
- responde más rápido;
- cuesta menos de ejecutar;
- es más fácil de integrar.

A veces el resultado honesto será: “el fine-tune no compensa; basta con buen prompting y RAG”. Eso también es aprender.

## Mentalidad experimental

Trabaja como si estuvieras haciendo un pequeño experimento:

```text
hipótesis → intervención → medición → comparación → decisión
```

Ejemplo:

```text
Hipótesis: un LoRA hará que el asistente SAP responda siempre con pasos numerados y pida datos faltantes.
Intervención: 300 ejemplos limpios y entrenamiento de 300 iteraciones.
Medición: 50 prompts no vistos, puntuados con criterios fijos.
Decisión: usar, repetir con más datos o descartar.
```

Esta mentalidad te protege de dos trampas: sobreentrenar porque “ya he invertido tiempo” y desplegar algo sin evidencia.

## 1. Casos de ejemplo

Elige un dominio acotado. Tres opciones con perfiles distintos:

### Variante A: Soporte técnico SAP

- **Dominio**: procedimientos internos de SAP ficticios.
- **Objetivo**: respuestas con pasos numerados, petición de datos faltantes, sin inventar transacciones.
- **Por qué funciona**: formato consistente, procedimientos estables, mucho ejemplo repetible.
- **Dataset mínimo**: 300 ejemplos de pares prompt-respuesta con sistema, usuario y asistente.

### Variante B: Asistente de nutrición

- **Dominio**: recomendaciones de nutrición general, no médicas.
- **Objetivo**: respuestas prudentes, estructuradas, con descargo de responsabilidad y preguntas de contexto.
- **Por qué funciona**: tono y formato controlables, conocimiento estable, casos claros de "no soy médico".
- **Dataset mínimo**: 250 ejemplos con disclaimer incluido.

### Variante C: Resumidor de documentos técnicos

- **Dominio**: resumir manuales, PDFs de producto o documentación interna.
- **Objetivo**: resúmenes estructurados (objetivo, pasos, advertencias) sin perder detalles críticos.
- **Por qué funciona**: combinación de [RAG](../03-Construir/01-RAG-local.md) para recuperar + LoRA para formato de resumen.
- **Dataset mínimo**: 200 ejemplos de documento → resumen estructurado.

No uses datos privados reales sin anonimizar. El objetivo es aprender el pipeline, no crear un producto médico/legal.

## 2. Arquitectura del pipeline

```text
modelo base 7B
  → dataset JSONL
  → LoRA en MLX
  → evaluación base vs adaptador
  → fuse
  → conversión/cuantización GGUF
  → Ollama o MLX server
  → agente/API local
  → evaluación final
```

Conceptos relacionados: [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md), [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md), [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md), [05-Modificacion-Pesos](../04-Adaptar/03-Merging-pruning-y-destilacion.md).

## 3. Elegir modelo base

Para Mac M2 24 GB:

- opción conservadora: `mlx-community/Qwen2.5-3B-Instruct-4bit`;
- opción recomendada si hay margen: `mlx-community/Qwen2.5-7B-Instruct-4bit`;
- alternativa rápida: Phi mini o Gemma pequeña.

No uses 14B para el primer proyecto final. Puede funcionar, pero ralentiza iteración. Primero demuestra el pipeline con 3B/7B.

## 4. Crear dataset

Estructura:

```bash
mkdir -p ~/ia-local/proyecto-final/data
mkdir -p ~/ia-local/proyecto-final/evals
```

Ejemplo para soporte técnico ficticio:

```bash
cat > ~/ia-local/proyecto-final/data/train.jsonl <<'EOF'
{"messages":[{"role":"system","content":"Eres un asistente de soporte SAP. Responde con pasos numerados y pide datos faltantes."},{"role":"user","content":"No puedo contabilizar una factura por error de centro de coste."},{"role":"assistant","content":"1. Comprueba que el centro de coste está activo.\n2. Verifica la sociedad y fecha contable.\n3. Revisa la derivación contable.\n4. Si el error sigue, envía código de error exacto y sociedad."}]}
{"messages":[{"role":"system","content":"Eres un asistente de soporte SAP. Responde con pasos numerados y pide datos faltantes."},{"role":"user","content":"El pedido aparece bloqueado."},{"role":"assistant","content":"1. Revisa el motivo de bloqueo en el pedido.\n2. Comprueba crédito, precio y disponibilidad.\n3. Valida si hay workflow pendiente.\n4. Indica número de pedido y mensaje exacto si necesitas escalar."}]}
EOF

cp ~/ia-local/proyecto-final/data/train.jsonl ~/ia-local/proyecto-final/data/valid.jsonl
```

Para un proyecto real, crea al menos:

- 200 ejemplos de entrenamiento;
- 30-50 de validación;
- 30-50 de evaluación no vistos.

## 5. Entrenar LoRA

```bash
cd ~/ia-local/curso
source .venv/bin/activate

mlx_lm.lora \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --train \
  --data ~/ia-local/proyecto-final/data \
  --iters 300 \
  --batch-size 1 \
  --lora-layers 16 \
  --learning-rate 1e-5 \
  --adapter-path ~/ia-local/proyecto-final/adapters/sap-lora
```

Si usas 7B y ves presión de memoria, reduce ejemplos largos o vuelve a 3B.

## 6. Probar modelo base vs adaptador

Base:

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --prompt "No puedo contabilizar una factura por error de centro de coste." \
  --max-tokens 180 \
  --temp 0.2
```

Adaptado:

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --adapter-path ~/ia-local/proyecto-final/adapters/sap-lora \
  --prompt "No puedo contabilizar una factura por error de centro de coste." \
  --max-tokens 180 \
  --temp 0.2
```

Evalúa si realmente sigue el formato y pide datos faltantes.

## 7. Fusionar

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --adapter-path ~/ia-local/proyecto-final/adapters/sap-lora \
  --save-path ~/ia-local/proyecto-final/fused/sap-qwen3b
```

Sirve con MLX:

```bash
mlx_lm.server \
  --model ~/ia-local/proyecto-final/fused/sap-qwen3b \
  --host 127.0.0.1 \
  --port 8081
```

Prueba:

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"sap-qwen3b","messages":[{"role":"user","content":"Tengo un bloqueo en un pedido."}],"temperature":0.2}' | jq
```

## 8. Exportar a GGUF y Ollama

Si necesitas Ollama:

1. exporta/fusiona a formato Hugging Face si tu versión de MLX lo permite;
2. convierte a GGUF con llama.cpp;
3. cuantiza a Q4_K_M;
4. crea un Modelfile.

Ejemplo de Modelfile con GGUF final:

```bash
cat > ~/ia-local/proyecto-final/Modelfile <<'EOF'
FROM ./sap-qwen3b-Q4_K_M.gguf
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
SYSTEM "Eres un asistente de soporte SAP. Responde con pasos numerados, sé prudente y pide datos faltantes."
EOF

cd ~/ia-local/proyecto-final
ollama create sap-local -f Modelfile
ollama run sap-local "Tengo un pedido bloqueado."
```

Si la exportación MLX→GGUF no está clara en tu versión, sirve con `mlx_lm.server`. Es una alternativa válida en Mac.

## 9. Conectar a un agente local

Cualquier cliente OpenAI-compatible puede apuntar al servidor local:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8081/v1"
export OPENAI_API_KEY="local"
```

Para agentes, usa prompts cortos y herramientas externas. No metas toda la base de conocimiento en el fine-tune.

## 10. Eval comparativo

Crea `~/ia-local/proyecto-final/evals/prompts.jsonl`:

```json
{"id":"sap-001","prompt":"No puedo contabilizar una factura por error de centro de coste.","criterios":["pasos numerados","pide código de error","no inventa transacciones"]}
```

Eval manual mínima:

| ID | Base | Fine-tune | Ganador | Motivo |
|---|---|---|---|---|
| sap-001 | 2/5 | 4/5 | fine-tune | mejor formato |

No declares éxito si solo mejora prompts vistos. Prueba casos nuevos.

## Entregable final

Tu entregable debe incluir:

- modelo base;
- dataset y tamaño;
- comandos usados;
- adaptador o modelo fusionado;
- endpoint local;
- tabla base vs fine-tune;
- limitaciones.

## Profundización: diseño de un experimento útil

El proyecto final debe responder a una pregunta concreta, no “hacer un fine-tune porque sí”. Formula el objetivo como una hipótesis falsable.

Ejemplos:

```text
Mala hipótesis: el modelo será mejor.
Buena hipótesis: el modelo reducirá respuestas sin pasos numerados de 60% a menos de 15% en 50 casos de soporte.
```

Cuanto más concreta la hipótesis, más fácil decidir si el proyecto funcionó.

### Separar tres fuentes de mejora

Cuando un sistema mejora, puede deberse a:

1. mejor prompt;
2. mejor contexto/herramientas;
3. pesos/adaptador mejor ajustados.

Si cambias las tres cosas a la vez, no sabrás qué aportó valor. En el proyecto final, compara en etapas:

```text
modelo base + prompt simple
modelo base + prompt mejorado
modelo base + contexto/herramientas
modelo con LoRA + mismo contexto
modelo cuantizado final
```

### Diseño de evals pequeñas pero útiles

No necesitas un benchmark académico enorme. Necesitas casos representativos. Una eval útil para un asistente de dominio puede tener:

- 10 casos fáciles;
- 20 casos normales;
- 10 casos ambiguos;
- 10 casos donde debe decir “no sé” o pedir datos;
- 5 casos fuera de dominio.

Esto detecta si el modelo solo aprendió a sonar convincente.

### Criterios de aceptación

Define criterios antes de mirar resultados. Por ejemplo:

```text
Acepto el modelo si:
- 80% de respuestas siguen formato de pasos.
- 90% piden dato faltante cuando el prompt es ambiguo.
- 0 respuestas inventan transacciones inexistentes.
- Latencia media es aceptable en el Mac.
```

Sin criterios previos, tenderás a justificar el resultado que ya obtuviste.

### Decisión final: desplegar, iterar o descartar

El final del proyecto no siempre es desplegar. Puede ser:

- **Desplegar**: mejora clara y coste aceptable.
- **Iterar**: mejora parcial; faltan datos o mejor eval.
- **Descartar LoRA**: prompt/RAG bastan o el modelo base es insuficiente.
- **Subir a cloud**: la máquina local limita demasiado.

Un buen curso enseña también cuándo no usar una técnica.

## Ejercicio práctico

Completa el pipeline con 3B o 7B. Si algo falla por memoria:

1. reduce modelo;
2. reduce longitud de ejemplos;
3. baja iteraciones;
4. usa cloud;
5. documenta el bloqueo y la alternativa.

## Recursos

- MLX LM: https://github.com/ml-explore/mlx-lm
- Ollama Modelfile: https://docs.ollama.com/modelfile
- llama.cpp conversion: https://github.com/ggml-org/llama.cpp
- LangGraph: https://langchain-ai.github.io/langgraph/
- Model Context Protocol: https://modelcontextprotocol.io/

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Evaluación y monitorización de calidad](../05-LLMOps/12-Evaluacion-y-calidad-en-produccion.md) · [Índice](../README.md) · [P1 - Proyecto - Motor de inferencia desde cero →](02-Motor-de-inferencia-desde-cero.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
