---
tags:
  - curso/ia-local
  - rag
  - embeddings
  - vector-db
curso: IA-Local-de-Cero-a-Produccion
modulo: '07'
estado: completo
---

# 07 - RAG local: pipelines con contexto

<!-- CURSO_NAV_TOP -->
[← 03 - Cuantización y reducción de memoria](../02-Uso-local/02-Cuantizacion-y-formatos.md) · [Índice](../README.md) · [08 - Agentes locales: LLM + herramientas →](02-Agentes-locales-y-MCP.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Linux, Windows y macOS
> El pipeline Python y Ollama funcionan igual en los tres sistemas. Solo cambia la activación del entorno y algún comando de terminal; consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).

> [!goals] Objetivos de aprendizaje
> - Entender por qué RAG es la primera herramienta que debes probar antes de pensar en fine-tuning.
> - Comprender embeddings, similitud coseno y por qué la recuperación semántica no es una búsqueda por palabras.
> - Instalar y usar modelos de embeddings locales (`nomic-embed-text`, `bge-m3`, `sentence-transformers`).
> - Montar una base vectorial local con ChromaDB o sqlite-vec.
> - Construir un pipeline RAG completo en Python: cargar, chunkar, embeddar, almacenar, recuperar e inyectar en el prompt de un LLM local.
> - Medir si el RAG funciona: precisión de recuperación, calidad de respuesta, reducción de alucinaciones.
> - Conocer estrategias avanzadas: chunking con overlap, reranking y cuándo aplicarlas.


---

## Contexto pedagógico: por qué RAG antes que fine-tuning

Cuando un LLM no sabe algo, el reflejo equivocado es "voy a fine-tunearlo". Casi siempre es un error.

El conocimiento tiene dos componentes distintos:

1. **Capacidad** — razonar, resumir, seguir instrucciones, escribir código. Eso sí vive en los pesos y se afina con fine-tuning (ver [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md)).
2. **Conocimiento** — hechos, documentos internos, API specs, normas de la empresa, fechas, precios. Eso **no** debe vivir en los pesos.

RAG (Retrieval-Augmented Generation) resuelve el segundo: traes el contexto relevante en el momento de la consulta y lo metes en el prompt. El modelo no aprende nada nuevo; solo lee.

### RAG vs LoRA: mentalidad comparada

| Dimensión | RAG | LoRA / Fine-tuning |
|---|---|---|
| Conocimiento cambiante | Ideal: actualizas la base vectorial | Pésimo: hay que reentrenar |
| Coste | CPU/embeddings + almacenamiento | GPU + datos etiquetados + tiempo |
| Transparencia | Alta: ves qué chunks se recuperaron | Baja: el conocimiento está difuso en los pesos |
| Actualización | Añadir/borrar documentos en segundos | Reentrenar el adaptador |
| Alucinación | Reducible: el modelo cita la fuente | Difícil de controlar |
| Cuándo | Documentos, FAQs, manuales, APIs | Estilo, tono, formato, tarea nueva |

Regla práctica: **si la información cambia más rápido de lo que tardas en entrenar un LoRA, usa RAG.** Si lo que quieres es que el modelo hable como tu marca o produzca JSON con un esquema concreto, entonces fine-tuning.

RAG y fine-tuning no son excluyentes: un LoRA de estilo + RAG de conocimiento es un patrón común y potente.

---

## Profundización: embeddings, similitud y recuperación

### ¿Qué es un embedding?

Un embedding es un vector de números (típicamente 384, 768 o 1024 dimensiones) que representa el **significado** de un texto. Textos con significado similar caen cerca en el espacio vectorial.

```
"el gato duerme en el sofá"        → [0.12, -0.44, 0.88, ...]
"el felino descansa en el sillón"  → [0.11, -0.41, 0.85, ...]  # muy cerca
"la bolsa cae un 3%"               → [-0.33, 0.72, -0.10, ...] # lejos
```

El modelo de embeddings se entrena para que frases con el mismo sentido produzcan vectores cercanos, aunque no compartan ninguna palabra.

### Distancia coseno

La métrica estándar no es la distancia euclídea sino la **similitud coseno**: el ángulo entre los vectores, ignorando su magnitud.

```
sim(A, B) = (A · B) / (||A|| · ||B||)
```

- 1.0 = mismo sentido
- 0.0 = ortogonal (sin relación)
- -1.0 = opuesto

Los modelos de embeddings modernos suelen producir vectores ya normalizados, así que el producto escalar basta.

### Por qué recuperar no es buscar por palabras

Una búsqueda por palabras (BM25, LIKE en SQL) encuentra "contrato" cuando buscas "contrato". No encuentra "acuerdo comercial" cuando buscas "contrato de servicios". RAG sí, porque compara significados, no tokens.

Además, los embeddings son tolerantes a:
- Sinónimos y paráfrasis
- Distintos idiomas (modelos multilingües como `bge-m3`)
- Errores tipográficos leves
- Preguntas vs afirmaciones ("¿cuál es la política de devoluciones?" vs "política de devoluciones")

RAG no sustituye a BM25; lo complementa. Los sistemas serios usan **búsqueda híbrida**: BM25 para palabras exactas + embeddings para semántica, fusionadas con RRF (Reciprocal Rank Fusion).

### Chunking

No puedes embeddar un PDF de 200 páginas entero: el modelo de embeddings tiene un límite de tokens (típicamente 512) y un vector único pierde detalle. Hay que dividir el documento en **chunks**.

Trade-offs de tamaño de chunk:

| Chunk pequeño (128-256 tokens) | Chunk grande (512-1024 tokens) |
|---|---|
| Recuperación precisa, poco contexto | Más contexto, más ruido |
| El LLM puede no tener suficiente | El LLM puede perderse |
| Más vectores, más almacenamiento | Menos vectores, más barato |
| Mejor para datos fragmentados (FAQs) | Mejor para prosa continua (manuales) |

No hay un número mágico. Empieza con 512 tokens y ajusta según tu evaluación (ver sección 4).

---

## 1. Modelos de embeddings locales

### Opción A: Ollama (la más simple)

Ollama (ver [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md)) sirve modelos de embeddings con la misma API que los LLM.

```bash
# Descargar modelos de embeddings
ollama pull nomic-embed-text    # 137M, 768 dim, rápido, muy usado
ollama pull bge-m3              # 568M, 1024 dim, multilingüe, denso + sparse + colbert
```

Generar un embedding desde macOS:

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "nomic-embed-text",
  "input": "la memoria RAM es volátil"
}'
```

Respuesta:

```json
{ "embeddings": [[0.012, -0.044, 0.088, "..."]] }
```

En PowerShell puedes usar `Invoke-RestMethod` como en la primera práctica, o saltar directamente al ejemplo Python siguiente.

### Opción B: sentence-transformers (Python nativo)

```bash
uv init rag-demo
cd rag-demo
uv add sentence-transformers
```

```python
# embed_demo.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # 33M, 384 dim, muy rápido en M2
# o para multilingüe: "BAAI/bge-m3"

texts = [
    "el gato duerme en el sofá",
    "el felino descansa en el sillón",
    "la bolsa cae un tres por ciento",
]

vecs = model.encode(texts, normalize_embeddings=True)

# Similitud coseno = producto escalar si están normalizados
import numpy as np
sim = vecs @ vecs.T
print(np.round(sim, 2))
# [[1.   0.82 0.12]   <- gato vs felino: alta
#  [0.82 1.   0.09]
#  [0.12 0.09 1.  ]]  <- bolsa vs gato: baja
```

### Opción C: Ollama desde Python

```python
# embed_ollama.py
import requests

def embed(text: str, model: str = "nomic-embed-text") -> list[float]:
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": model, "input": text},
    )
    r.raise_for_status()
    return r.json()["embeddings"][0]

v = embed("la memoria RAM es volátil")
print(len(v), v[:5])  # 768 [0.012, -0.044, ...]
```

**Recomendación:** empieza con `nomic-embed-text` vía Ollama. Si necesitas multilingüe o más calidad, prueba `bge-m3`. Para experimentar en Python puro sin servidor, `sentence-transformers` funciona en Linux, Windows y macOS.

---

## 2. Base vectorial local

### Opción A: ChromaDB (la más usada en tutoriales)

```bash
uv add chromadb
```

```python
# chroma_demo.py
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Crear (o abrir) una colección
collection = client.get_or_create_collection(
    name="docs_demo",
    metadata={"hnsw:space": "cosine"},  # distancia coseno
)

# Insertar documentos
collection.add(
    ids=["d1", "d2", "d3"],
    documents=[
        "La memoria RAM es volátil: pierde su contenido al apagar.",
        "El SSD conserva los datos sin alimentación eléctrica.",
        "Python es un lenguaje interpretado y de tipado dinámico.",
    ],
    metadatas=[
        {"tema": "hardware", "fuente": "apuntes.md"},
        {"tema": "hardware", "fuente": "apuntes.md"},
        {"tema": "software", "fuente": "intro.py"},
    ],
)

# ChromaDB embedda automáticamente con su modelo por defecto.
# Para usar Ollama/nomic, pasa un embedding_function (ver sección 3).

# Consultar
results = collection.query(
    query_texts=["¿qué pasa con la memoria al apagar el ordenador?"],
    n_results=2,
)

for doc, dist, meta in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0],
):
    print(f"[{dist:.3f}] {meta['tema']:8} {doc}")
```

Salida esperada:

```
[0.12] hardware La memoria RAM es volátil: pierde su contenido al apagar.
[0.45] hardware El SSD conserva los datos sin alimentación eléctrica.
```

### Opción B: sqlite-vec (más ligero, sin servidor)

```bash
uv add sqlite-vec
```

```python
# sqlite_vec_demo.py
import sqlite3
import sqlite_vec
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

db = sqlite3.connect(":memory:")  # o "vec.db" para persistir
db.enable_load_extension(True)
sqlite_vec.load(db)

db.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS docs
    USING vec0(embedding float[384], content text)
""")

docs = [
    "La memoria RAM es volátil.",
    "El SSD conserva los datos sin corriente.",
    "Python es interpretado y de tipado dinámico.",
]

for d in docs:
    v = model.encode(d, normalize_embeddings=True).tolist()
    db.execute("INSERT INTO docs(embedding, content) VALUES (?, ?)", (v, d))

# Consultar
q = "¿qué pasa con la RAM al apagar?"
qv = model.encode(q, normalize_embeddings=True).tolist()

rows = db.execute("""
    SELECT content, distance
    FROM docs
    WHERE embedding MATCH ?
    ORDER BY distance
    LIMIT 2
""", (qv,)).fetchall()

for content, dist in rows:
    print(f"[{dist:.3f}] {content}")
```

**Recomendación:** ChromaDB para prototipos rápidos y si te gusta su API. sqlite-vec si quieres cero dependencias extra y almacenamiento en un único fichero `.db`. Ambos corren nativamente en Apple Silicon.

---

## 3. Pipeline RAG completo

Pipeline end-to-end: carga de documentos → chunking → embeddings → almacenamiento → recuperación → generación con LLM local.

### Dependencias

```bash
uv init rag_pipeline
cd rag_pipeline
uv add chromadb sentence-transformers requests
# Ollama corriendo en otra terminal: ollama serve
# Modelo LLM descargado: ollama pull llama3.2:3b   (o qwen2.5:7b si tienes margen)
```

### Código

```python
# rag_pipeline.py
"""Pipeline RAG completo: documentos → chunks → vector DB → recuperación → LLM local."""
from __future__ import annotations

import os
import re
import json
import textwrap
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# 0. Configuración
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384 dim, rápido en M2
LLM_MODEL   = "llama3.2:3b"              # vía Ollama; también vale qwen2.5:7b
OLLAMA_URL  = "http://localhost:11434"
CHUNK_SIZE  = 400   # caracteres; para producción mide en tokens
CHUNK_OVERLAP = 80  # solapamiento entre chunks
TOP_K        = 4
DB_PATH      = "./chroma_rag"
DOCS_DIR     = "./docs"   # pon aquí tus .txt o .md


# ---------------------------------------------------------------------------
# 1. Cargar documentos
# ---------------------------------------------------------------------------
def load_documents(docs_dir: str) -> list[dict]:
    docs = []
    for path in sorted(Path(docs_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append({"id": path.stem, "text": text, "source": str(path)})
    for path in sorted(Path(docs_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({"id": path.stem, "text": text, "source": str(path)})
    if not docs:
        # datos de ejemplo si no hay documentos
        docs = [
            {"id": "demo1", "source": "demo",
             "text": "La memoria RAM es volátil: pierde su contenido al apagar el equipo. "
                     "El SSD, en cambio, conserva los datos sin alimentación eléctrica."},
            {"id": "demo2", "source": "demo",
             "text": "Python es un lenguaje interpretado, de tipado dinámico y multiparadigma. "
                     "Soporta programación funcional, orientada a objetos y procedural."},
            {"id": "demo3", "source": "demo",
             "text": "Ollama permite ejecutar LLMs localmente en Linux, Windows y macOS. "
                     "Expone una API REST en el puerto 11434."},
        ]
    return docs


# ---------------------------------------------------------------------------
# 2. Chunking (por caracteres, con solapamiento)
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap  # avanza dejando solapamiento
    return chunks


def chunk_documents(docs: list[dict]) -> list[dict]:
    chunks = []
    for doc in docs:
        for i, c in enumerate(chunk_text(doc["text"])):
            chunks.append({
                "id": f"{doc['id']}_{i}",
                "text": c,
                "source": doc["source"],
            })
    return chunks


# ---------------------------------------------------------------------------
# 3. Embeddings + almacenamiento en ChromaDB
# ---------------------------------------------------------------------------
def build_or_load_db(chunks: list[dict], embed_model_name: str, db_path: str):
    embedder = SentenceTransformer(embed_model_name)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="rag_corpus",
        metadata={"hnsw:space": "cosine"},
    )

    # Si la colección ya tiene datos, no reinsertamos
    if collection.count() == 0 and chunks:
        texts  = [c["text"] for c in chunks]
        ids    = [c["id"] for c in chunks]
        metas  = [{"source": c["source"]} for c in chunks]
        vecs   = embedder.encode(texts, normalize_embeddings=True).tolist()
        collection.add(ids=ids, embeddings=vecs, documents=texts, metadatas=metas)
        print(f"Indexados {len(chunks)} chunks en {db_path}")
    else:
        print(f"Colección ya tiene {collection.count()} chunks (reutilizando)")

    return collection, embedder


# ---------------------------------------------------------------------------
# 4. Recuperar top-k
# ---------------------------------------------------------------------------
def retrieve(collection, embedder, query: str, top_k: int = TOP_K) -> list[dict]:
    qv = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=qv, n_results=top_k)
    out = []
    for doc, dist, meta in zip(
        res["documents"][0], res["distances"][0], res["metadatas"][0]
    ):
        out.append({"text": doc, "distance": dist, "source": meta.get("source", "?")})
    return out


# ---------------------------------------------------------------------------
# 5. Generar respuesta con el LLM local (Ollama)
# ---------------------------------------------------------------------------
def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] ({c['source']}) {c['text']}" for i, c in enumerate(context_chunks)
    )
    return textwrap.dedent(f"""\
        Eres un asistente técnico. Responde la pregunta usando ÚNICAMENTE
        el contexto proporcionado. Si el contexto no contiene la respuesta,
        di "No tengo información suficiente en el contexto".

        Contexto:
        {context}

        Pregunta: {query}

        Respuesta:""")


def generate(prompt: str, model: str = LLM_MODEL) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


# ---------------------------------------------------------------------------
# 6. Pipeline completo
# ---------------------------------------------------------------------------
def rag_query(question: str, docs_dir: str = DOCS_DIR):
    docs   = load_documents(docs_dir)
    chunks = chunk_documents(docs)
    collection, embedder = build_or_load_db(chunks, EMBED_MODEL, DB_PATH)

    retrieved = retrieve(collection, embedder, question)
    print("\n=== Contexto recuperado ===")
    for c in retrieved:
        print(f"  [{c['distance']:.3f}] {c['source']:20} {c['text'][:80]}...")

    prompt = build_prompt(question, retrieved)
    answer = generate(prompt)
    print("\n=== Respuesta del LLM ===")
    print(answer)
    return answer


if __name__ == "__main__":
    rag_query("¿Qué le pasa a la memoria RAM cuando apago el ordenador?")
    rag_query("¿Qué es Ollama y en qué puerto escucha?")
```

### Ejecución

```bash
# Terminal 1
ollama serve

# Terminal 2
mkdir -p docs
# (opcional) copia aquí tus .txt / .md
uv run rag_pipeline.py
```

En Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force docs
uv run python rag_pipeline.py
```

Salida esperada:

```
Indexados 6 chunks en ./chroma_rag

=== Contexto recuperado ===
  [0.08] demo                 La memoria RAM es volátil: pierde su contenido al apagar el equipo. El SSD, en camb...
  [0.31] demo                 La memoria RAM es volátil: pierde su contenido al apagar el equipo. El SSD, en camb...

=== Respuesta del LLM ===
La memoria RAM es volátil, por lo que pierde su contenido al apagar el equipo.
```

### Variantes

- **Usar `mlx_lm.server` en vez de Ollama:** cambia `generate()` para atacar a `http://localhost:8080/v1/chat/completions` (API compatible con OpenAI). Ver [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md) para levantar el servidor.
- **Usar `bge-m3` vía Ollama para embeddings:** sustituye `SentenceTransformer` por la función `embed_ollama()` de la sección 1 y pasa los vectores a `collection.add(..., embeddings=...)`.
- **Modelo más grande:** si tienes margen de memoria, `qwen2.5:7b` o `llama3.1:8b` dan mejores respuestas. La cuantización (ver [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md)) mantiene el consumo asumible en 24 GB.

---

## 4. Evaluación del RAG

Un RAG sin evaluación es una caja negra. Necesitas métricas para dos cosas distintas: **¿recupero bien?** y **¿respondo bien?**

### Evaluación de la recuperación

Construye un conjunto de preguntas de prueba con su **chunk correcto esperado** (gold answer). Para cada pregunta, mide si el chunk correcto aparece en el top-k.

```python
# eval_retrieval.py
def retrieval_precision_at_k(collection, embedder, test_cases: list[dict], k: int = 4):
    hits = 0
    for case in test_cases:
        retrieved = retrieve(collection, embedder, case["question"], top_k=k)
        retrieved_ids = [r["source"] for r in retrieved]  # o el id del chunk
        if case["expected_source"] in retrieved_ids:
            hits += 1
    return hits / len(test_cases)

test_cases = [
    {"question": "¿La RAM es volátil?",
     "expected_source": "demo"},
    {"question": "¿Qué puerto usa Ollama?",
     "expected_source": "demo"},
]

p = retrieval_precision_at_k(collection, embedder, test_cases, k=4)
print(f"Precision@4: {p:.2f}")  # 1.0 = perfecto
```

Métricas útiles:

- **Precision@k** — qué fracción de consultas tienen el chunk relevante en el top-k.
- **Recall@k** — qué fracción de chunks relevantes aparecen.
- **MRR (Mean Reciprocal Rank)** — premisa que el chunk correcto salga en primer lugar.

### Evaluación de la respuesta

- **Factualidad** — ¿la respuesta es correcta según el contexto?
- **Fidelidad (faithfulness)** — ¿la respuesta se apoya en el contexto o inventa?
- **Reducción de alucinación** — compara respuestas del LLM con y sin RAG sobre las mismas preguntas.

Un protocolo simple:

1. Define 10-20 preguntas con respuesta conocida.
2. Ejecuta el RAG y guarda respuestas + chunks recuperados.
3. Etiqueta manualmente (o con un LLM como juez, ver [Evaluacion-LLMs-Local](../07-Anexos/A-Evaluacion-local-sin-autoengano.md)): correcto / incorrecto / alucinado.
4. Ajusta chunk size, top-k, prompt y vuelve a medir.

Si la recuperación es buena pero la respuesta es mala → problema de prompt o de modelo.
Si la recuperación es mala → problema de embeddings, chunking o calidad de los documentos.

---

## Profundización: chunking, overlap y reranking

### Estrategias de chunking

| Estrategia | Cuándo | Notas |
|---|---|---|
| Por tamaño fijo (caracteres/tokens) | Prototipos, prosa genérica | Simple, puede cortar frases a medias |
| Por párrafos / saltos de línea | Manuales, artículos | Respeta unidades naturales |
| Por estructura (Markdown headings) | Documentación técnica | Cada sección es un chunk |
| Por frases (sentence-aware) | Legal, normativo | Evita cortar ideas |
| Recursivo (LangChain `RecursiveCharacterTextSplitter`) | Caso general | Intenta párrafo → frase → palabra |

### Overlap (solapamiento)

Si chunk_size=400 y overlap=80, cada chunk empieza 80 caracteres antes del final del anterior. Sirve para no perder contexto en los cortes: una idea que cae justo en la frontera entre dos chunks aparece completa en al menos uno.

- Sin overlap: barato, pero pierdes ideas partidas.
- Con overlap: más vectores (más coste), menos pérdida.
- Regla: overlap = 10-20% del chunk size.

### Reranking

La recuperación vectorial es rápida pero imperfecta: a veces trae chunks que son semánticamente cercanos pero poco relevantes. Un **reranker** es un modelo (típicamente cross-encoder) que reordena los top-k candidatos con mayor precisión.

```
Consulta → embeddings → recuperar top-20 (rápido, aproximado)
                      → reranker cross-encoder → top-4 finales (lento, preciso)
```

Cuándo hace falta reranking:

- Tienes muchos documentos y el top-k trae ruido.
- La diferencia entre chunks relevantes y no relevantes es sutil.
- Tu base de usuarios nota que "a veces acierta, a veces no".

Cuándo **no** hace falta:

- Corpus pequeño (<1000 chunks): la recuperación simple ya es buena.
- Prototipo o MVP: añade complejidad sin retorno claro.
- Latencia crítica: el reranker añade 50-200ms por consulta.

Reranker local típico: `BAAI/bge-reranker-v2-m3` vía `sentence-transformers`:

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs = [(query, c["text"]) for c in retrieved]
scores = reranker.predict(pairs)
retrieved = [r for _, r in sorted(zip(scores, retrieved), key=lambda x: -x[0])]
```

---

## Ejercicio práctico

**Objetivo:** monta un RAG sobre tus propios apuntes y evalúalo.

1. **Prepara datos.** Crea `./docs/` y mete 5-10 archivos `.md` o `.txt` (apuntes, manuales, FAQs).
2. **Levanta el LLM.** `ollama serve` y `ollama pull llama3.2:3b`.
3. **Ejecuta el pipeline** de la sección 3 con `uv run rag_pipeline.py`. Verifica que responde preguntas sobre tus documentos.
4. **Define 5 preguntas** de prueba con la respuesta esperada.
5. **Mide precisión de recuperación** con `retrieval_precision_at_k`. Si es <0.8:
   - Prueba chunk_size 200 y 800. Compara.
   - Cambia `nomic-embed-text` por `bge-m3`. Compara.
   - Añade reranking. Compara.
6. **Mide calidad de respuesta**: 5 respuestas correctas de 5 = OK. Si hay alucinaciones, refuerza el prompt ("responde ÚNICAMENTE con el contexto").
7. **Sube el modelo** a `qwen2.5:7b` (ver [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md)) y compara calidad.
8. **Documenta tus resultados** en una tabla dentro de Obsidian.

Entregable: repositorio con `rag_pipeline.py`, `eval_retrieval.py`, `docs/` y una tabla de resultados.

---

## Recursos

- **ChromaDB docs** — https://docs.trychroma.com/
- **sqlite-vec** — https://github.com/asg017/sqlite-vec
- **sentence-transformers** — https://www.sbert.net/
- **BGE embeddings (BAAI)** — https://huggingface.co/BAAI/bge-m3
- **nomic-embed-text (Ollama)** — https://ollama.com/library/nomic-embed-text
- **Ollama embeddings API** — https://ollama.com/blog/embedding-models
- **LangChain RecursiveCharacterTextSplitter** — https://python.langchain.com/docs/how_to/recursive_text_splitter/
- **RAGFromScratch (LangChain)** — https://github.com/langchain-ai/rag-from-scratch
- **Evaluación de LLMs y RAG (Ragas)** — https://docs.ragas.io/
- **BGE reranker** — https://huggingface.co/BAAI/bge-reranker-v2-m3

---

> [!note] Relacionado
> Este módulo asume que dominas la inferencia local ([02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md)) y la cuantización ([03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md)). Para cambiar estilo o formato del LLM, ver [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md). Para evaluar sistemáticamente, [Evaluacion-LLMs-Local](../07-Anexos/A-Evaluacion-local-sin-autoengano.md). El siguiente paso es dar agencia al modelo: [08-Agentes-Locales](02-Agentes-locales-y-MCP.md).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← 03 - Cuantización y reducción de memoria](../02-Uso-local/02-Cuantizacion-y-formatos.md) · [Índice](../README.md) · [08 - Agentes locales: LLM + herramientas →](02-Agentes-locales-y-MCP.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
