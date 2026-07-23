---
tags:
  - curso/ia-local
  - agentes
  - tools
  - mcp
curso: IA-Local-de-Cero-a-Produccion
modulo: '08'
estado: completo
---

# 08 - Agentes locales: LLM + herramientas

<!-- CURSO_NAV_TOP -->
[← 07 - RAG local: pipelines con contexto](01-RAG-local.md) · [Índice](../README.md) · [IA multimodal local: imagen, texto y audio →](03-IA-multimodal-local.md)
<!-- /CURSO_NAV_TOP -->



> [!info] Windows y macOS
> El código Python y la API de Ollama son comunes. En este capítulo las tools quedan limitadas a una carpeta de laboratorio: no damos al modelo una shell abierta ni acceso a todo el disco.

> [!goals] Objetivos de aprendizaje
> - Entender por qué un LLM aislado no puede actuar sobre el mundo y qué aporta un agente.
> - Implementar un bucle ReAct mínimo con Ollama y funciones Python.
> - Usar LangGraph para construir agentes locales con estado.
> - Montar un servidor MCP local y conectarlo a un cliente.
> - Distinguir cuándo un agente aporta valor frente a un prompt directo.


---

## Contexto pedagógico: por qué un LLM solo no basta

Un LLM es, por diseño, una función de texto a texto: recibe tokens y devuelve tokens. No puede buscar en la web, ejecutar código, leer un fichero de tu disco, ni llamar a una API REST. Sus "conocimientos" están congelados en el momento del entrenamiento y limitados por la ventana de contexto. Cuando le pides "resume el PDF que tengo en Downloads", el modelo no puede acceder a ese archivo por sí mismo.

Un **agente** resuelve esto añadiendo un bucle de control alrededor del LLM: el modelo decide qué herramienta usar, el sistema la ejecuta, y el resultado se reinyecta en el contexto para que el modelo continúe. El LLM actúa como razonador y planificador; las herramientas son sus manos.

### Comparativa: LLM solo vs LLM + tools vs LLM + RAG + tools

| Enfoque | Qué puede | Qué no puede | Cuándo usarlo |
|---|---|---|---|
| **LLM solo** | Responder con conocimiento paramétrico, razonar sobre el prompt, resumir, traducir | Acceder a datos externos, ejecutar acciones, consultar tiempo real | Preguntas generales, brainstorming, redacción |
| **LLM + tools** | Llamar APIs, leer ficheros, ejecutar código, buscar en web | Recuperar knowledge base propia eficientemente | Automatización, asistentes con acciones, scripting inteligente |
| **LLM + RAG + tools** | Todo lo anterior + recuperar documentos de una base vectorial local | (su principal coste es complejidad y latencia) | Asistentes sobre documentación privada con capacidad de acción |

> [!note] Relación con módulos previos
> El módulo [07-RAG-Local](01-RAG-local.md) cubrió la recuperación. Aquí añadimos el bucle de acción. En la práctica, un agente completo combina ambas: RAG es una tool más dentro del bucle ReAct.

---

## Profundización: el bucle ReAct

**ReAct** (Reasoning + Acting) es el patrón de referencia para agentes desde 2022. El ciclo es:

```
1. Reason  → El LLM piensa qué hacer next (Thought)
2. Act     → El LLM emite un tool call (Action)
3. Observe → El sistema ejecuta la tool y devuelve el resultado (Observation)
4. Repeat  → Volver a 1 hasta que el LLM emite una respuesta final
```

El modelo no ejecuta nada: **propone** una llamada, el runtime la ejecuta. La salida del LLM puede ser texto libre (formato ReAct clásico con `Thought:` / `Action:` / `Observation:`) o, en modelos modernos con soporte nativo, un **tool call** estructurado en JSON que el parser extrae sin ambigüedad.

### ¿Por qué los agentes iteran?

Porque una sola llamada a una tool rara vez resuelve el problema. Por ejemplo, para "busca el PIB de España 2024 y guárdalo en un fichero", el agente necesita: (1) buscar el dato, (2) leer el resultado, (3) escribir el fichero. Cada paso depende del anterior. El bucle permite encadenar herramientas sin que el humano defina el flujo a mano.

### ¿Qué es un tool call?

Un **tool call** es una instrucción estructurada que el LLM emite en su salida indicando:

- **nombre** de la función a ejecutar
- **argumentos** (JSON con los parámetros)

El runtime intercepta esa instrucción, la valida contra el esquema declarado de la tool, ejecuta la función Python correspondiente, y reinyecta el resultado como un mensaje de rol `tool` en el siguiente turno.

Familias como Qwen, Llama, Mistral y Granite incluyen variantes con tool calling. Comprueba la etiqueta `tools` en la [biblioteca de Ollama](https://ollama.com/search): compartir familia no garantiza que todos los tamaños o cuantizaciones se comporten igual.

### El problema de los bucles infinitos

Si el LLM nunca decide que la tarea está completa —o si una tool falla silenciosamente y el modelo reintenta lo mismo una y otra vez— el agente entra en un bucle infinito. Estrategias para evitarlo:

- **`max_iterations`** hard cap (típicamente 5–10).
- **Detección de llamadas repetidas** idénticas: si el mismo tool call se repite 2 veces seguidas, abortar.
- **Tool de parada explícita**: una tool `finish(result)` que el modelo debe llamar para terminar.
- **Timeout** global del bucle.

---

## 1. Herramientas básicas

Vamos a construir un agente mínimo con Ollama y Python puro, sin frameworks. Usaremos `qwen3.5:4b`; si el tag cambia, elige un modelo pequeño que indique soporte de tools.

### 1.1. Definir las tools

Cada tool es una función Python con un esquema JSON que describe su firma para el LLM.

```python
# agent_tools.py
import json
import urllib.request
import platform
from pathlib import Path

LAB_ROOT = Path.cwd().resolve()


def wikipedia_search(query: str, lang: str = "es") -> str:
    """Busca un resumen en Wikipedia y devuelve el extracto."""
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("extract", "[sin resultado]")
    except Exception as e:
        return f"[error wikipedia: {e}]"


def read_file(path: str) -> str:
    """Lee un fichero dentro del laboratorio y devuelve hasta 4000 caracteres."""
    p = (LAB_ROOT / path).resolve()
    if not p.is_relative_to(LAB_ROOT):
        return "[error: la ruta sale de la carpeta de laboratorio]"
    if not p.is_file():
        return f"[error: no existe {path}]"
    return p.read_text(encoding="utf-8", errors="replace")[:4000]


def list_markdown_files() -> str:
    """Lista Markdown dentro del laboratorio, sin salir de él."""
    files = sorted(p.relative_to(LAB_ROOT) for p in LAB_ROOT.rglob("*.md"))
    return "\n".join(map(str, files[:100])) or "[sin ficheros Markdown]"


def system_info() -> str:
    """Devuelve datos no sensibles del sistema, sin ejecutar una shell."""
    return f"{platform.system()} {platform.release()} · {platform.machine()}"


# Esquema que el LLM verá. Ollama usa formato OpenAI-compatible.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Busca un resumen de un término en Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Término a buscar"},
                    "lang": {"type": "string", "description": "Idioma, ej: es, en"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee el contenido de un fichero local.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Ruta al fichero"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_markdown_files",
            "description": "Lista ficheros Markdown dentro de la carpeta de laboratorio.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": "Devuelve sistema operativo y arquitectura.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Mapa nombre → función para dispatch
TOOL_MAP = {
    "wikipedia_search": wikipedia_search,
    "read_file": read_file,
    "list_markdown_files": list_markdown_files,
    "system_info": system_info,
}
```

### 1.2. Bucle ReAct con Ollama API

```python
# agent_loop.py
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"
MAX_ITER = 8

from agent_tools import TOOLS_SCHEMA, TOOL_MAP

SYSTEM_PROMPT = """Eres un agente local.
Tienes herramientas disponibles. Para responder a una pregunta:
1. Piensa si necesitas usar una herramienta.
2. Si la necesitas, llama a la herramienta indicando su nombre y argumentos.
3. Observa el resultado y decide si necesitas otra herramienta o ya puedes responder.
4. Cuando tengas la respuesta final, responde directamente al usuario en español.
Sé conciso."""

def call_ollama(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS_SCHEMA,
        "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

def run_agent(user_query: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    for i in range(MAX_ITER):
        print(f"\n--- Iteración {i+1} ---")
        resp = call_ollama(messages)
        msg = resp["message"]
        messages.append(msg)

        # ¿El modelo quiere llamar tools?
        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            # No hay tool calls → respuesta final
            print("\n[Respuesta final]:", msg["content"])
            return msg["content"]

        # Ejecutar cada tool call
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)
            print(f"  → tool: {fn_name}({fn_args})")
            result = TOOL_MAP[fn_name](**fn_args)
            print(f"  ← {result[:120]}...")
            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": str(result),
            })

    print("\n[Abortado: max_iteraciones alcanzado]")
    return None

if __name__ == "__main__":
    run_agent("¿Qué temas aparecen en el README.md de esta carpeta de laboratorio?")
```

> [!tip] Probarlo

```text
uv run python agent_loop.py
```

Verás cada iteración, qué tool llama el modelo y el resultado. El modelo decide el flujo, pero solo puede usar las funciones limitadas que has declarado.

---

## 2. LangGraph local

LangGraph (parte de LangChain) permite definir el bucle del agente como un **grafo de nodos** con estado explícito. Es más robusto que un `while` a mano y facilita paralelismo, checkpoints y human-in-the-loop.

### 2.1. Instalación

```bash
uv init agente-lg && cd agente-lg
uv add langchain-ollama langgraph
```

### 2.2. Configurar modelo local con tools

```python
# agente_langgraph.py
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Definir tools con el decorador @tool
@tool
def wikipedia_search(query: str) -> str:
    """Busca un resumen en Wikipedia (español)."""
    import urllib.request, json
    url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
    return data.get("extract", "[sin resultado]")

@tool
def read_file(path: str) -> str:
    """Lee un fichero dentro del laboratorio."""
    from pathlib import Path
    root = Path.cwd().resolve()
    p = (root / path).resolve()
    if not p.is_relative_to(root):
        return "[error: ruta fuera del laboratorio]"
    if not p.is_file():
        return f"[error: no existe {path}]"
    return p.read_text(errors="replace")[:4000]

@tool
def list_markdown_files() -> str:
    """Lista los Markdown de la carpeta de laboratorio."""
    from pathlib import Path
    root = Path.cwd().resolve()
    return "\n".join(str(p.relative_to(root)) for p in sorted(root.rglob("*.md"))[:100])

# Modelo local con tool calling nativo
llm = ChatOllama(
    model="qwen3.5:4b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

# create_react_agent monta el grafo ReAct por nosotros
agent = create_react_agent(llm, [wikipedia_search, read_file, list_markdown_files])

# Ejecutar
result = agent.invoke({
    "messages": [HumanMessage(content="Lista los Markdown del laboratorio y resume el primero.")]
})

# Imprimir solo el último mensaje
final = result["messages"][-1]
print(final.content)
```

### 2.3. Grafo personalizado (más control)

Si quieres definir los nodos a mano —por ejemplo para añadir validación o límite de iteraciones explícito—:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iter_count: int

def call_model(state):
    response = llm.bind_tools([wikipedia_search, read_file, list_markdown_files]).invoke(state["messages"])
    return {"messages": [response], "iter_count": state["iter_count"] + 1}

def should_continue(state):
    last = state["messages"][-1]
    if state["iter_count"] >= 8:
        return END
    if last.tool_calls:
        return "tools"
    return END

# Nodos
graph = StateGraph(AgentState)
graph.add_node("model", call_model)
graph.add_node("tools", tool_node)  # tool_node de langgraph.prebuilt

# Edges
graph.add_edge(START, "model")
graph.add_conditional_edges("model", should_continue, ["tools", END])
graph.add_edge("tools", "model")

app = graph.compile()
```

> [!note] Recursos
> La guía oficial de LangGraph: https://langchain-ai.github.io/langgraph/

---

## 3. MCP (Model Context Protocol)

MCP es un **protocolo abierto** (publicado por Anthropic en 2024) para estandarizar cómo un cliente LLM descubre y usa herramientas expuestas por un servidor. La idea: en lugar de programar tools en cada app, montas un **servidor MCP** que expone tools como recursos; cualquier cliente MCP (Claude Desktop, IDEs, tu agente) puede usarlos sin acoplamiento de código.

### 3.1. Por qué MCP en local

- **Desacoplamiento**: el servidor MCP no sabe nada del LLM; solo responde JSON-RPC.
- **Reutilización**: el mismo servidor sirve para Cursor, Claude Desktop, tu agente Python.
- **Estandarización**: esquema de tools auto-descrito vía `list_tools`.

### 3.2. Servidor MCP mínimo con una tool

```bash
uv add mcp
```

```python
# servidor_mcp.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agente-local")

@mcp.tool()
def read_file(path: str) -> str:
    """Lee un fichero dentro del laboratorio y devuelve hasta 4000 caracteres."""
    from pathlib import Path
    root = Path.cwd().resolve()
    p = (root / path).resolve()
    if not p.is_relative_to(root):
        return "[error: ruta fuera del laboratorio]"
    if not p.is_file():
        return f"[error: no existe {path}]"
    return p.read_text(errors="replace")[:4000]

@mcp.tool()
def list_markdown_files() -> str:
    """Lista Markdown dentro de la carpeta de laboratorio."""
    from pathlib import Path
    root = Path.cwd().resolve()
    files = sorted(p.relative_to(root) for p in root.rglob("*.md"))
    return "\n".join(map(str, files[:100])) or "[sin ficheros Markdown]"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 3.3. Probarlo

El servidor corre por **stdio** (sin red). Lo ejecuta el cliente y se comunican por pipes:

```bash
# Instalar el inspector oficial para testear
npx @modelcontextprotocol/inspector uv run servidor_mcp.py
```

Se abre una web en `http://localhost:5173` donde puedes ver las tools expuestas, inspeccionar sus esquemas y llamarlas manualmente.

### 3.4. Conectarlo desde Python

```python
# cliente_mcp.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command="uv",
        args=["run", "servidor_mcp.py"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            for t in tools.tools:
                print(f"Tool: {t.name} — {t.description}")

            # Llamar una tool
            result = await session.call_tool("list_markdown_files", {})
            print("Resultado:", result.content[0].text)

asyncio.run(main())
```

### 3.5. Conectarlo a un agente

LangChain y otros frameworks ya tienen adaptadores `langchain-mcp-adapters` que convierten las tools MCP en tools LangChain utilizables por `create_react_agent`. Así, un agente puede usar indistintamente tools locales y tools remotas expuestas vía MCP.

> [!tip] Recursos oficiales
> - Especificación: https://modelcontextprotocol.io/
> - SDK Python: https://github.com/modelcontextprotocol/python-sdk
> - Servidores de referencia: https://github.com/modelcontextprotocol/servers

---

## 4. Patrones y anti-patrones

### Cuántas tools

- **3–5 tools** es un sweet spot para modelos 7B–14B. Más allá, el modelo empieza a confundirse sobre cuál elegir.
- Modelos más grandes (32B+) pueden manejar 10–15 tools sin degradarse.
- Si necesitas más, **agrupa** por dominio y activa solo las relevantes en cada nodo del grafo.

### Cómo nombrar las tools

- Verbo + objeto: `read_file`, `search_wikipedia`, `list_markdown_files`.
- Descripción corta y orientada a acción: que el modelo entienda en una frase qué hace.
- El docstring **es** la documentación para el LLM: sé explícito sobre qué argumentos espera y qué devuelve.

### Errores comunes

- **Esquemas ambiguos**: si una tool acepta `query` y otra `search_query`, el modelo las confunde. Estandariza nombres.
- **Tools que fallan en silencio**: si una tool devuelve `""` o `None`, el modelo no sabe si fue éxito o fracaso. Devuelve `[error: ...]` siempre.
- **Demasiados argumentos opcionales**: el modelo genera basura. Si un arg es opcional, pon un default razonable.
- **Tools inseguras**: una shell arbitraria permite que una instrucción maliciosa lea, modifique o borre datos. Prefiere funciones concretas, rutas acotadas, argumentos validados, confirmación humana y un sandbox cuando sea necesario.
- **No acotar el contexto**: cada iteración añade mensajes. Sin poda, en 5 turnos tienes 20k tokens de contexto. Usa ventanas deslizantes o resúmenes.

### Cuándo un agente es excesivo

Un agente añade latencia (varias llamadas al LLM) y no determinismo. Si la tarea se resuelve con:

- **Un prompt directo** → no montas agente.
- **Un script Python con un prompt para extraer un campo** → no montas agente.
- **RAG con un prompt fijo** (ver [07-RAG-Local](01-RAG-local.md)) → no montas agente.

Un agente es útil cuando el flujo **no es predecible**: el número de pasos y qué herramienta usar en cada uno dependen de los resultados intermedios. Si puedes dibujar el flowchart antes de tiempo, usa un pipeline, no un agente.

> [!warning] Regla práctica
> Si puedes escribir el flujo como `prompt1 → tool1 → prompt2 → tool2`, no necesitas un bucle ReAct. Necesitas un pipeline. El agente es para cuando el próximo paso depende del resultado del anterior y no lo sabes a priori.

---

## Profundización: memoria, estado y concurrencia

### Estado entre turnos

Un agente conversacional necesita recordar turnos previos. Estrategias:

1. **Historial completo**: pasar todos los mensajes cada turno. Simple, pero el contexto crece linealmente. Funciona hasta ~20k tokens con Qwen 32k de ventana.
2. **Ventana deslizante**: mantener solo los últimos N mensajes. Pierde contexto distante.
3. **Resumen acumulado**: cada K turnos, un LLM resume los anteriores en un mensaje `system` y se descartan los originales. Buena relación coste/memoria.
4. **Memoria externa**: guardar hechos extraídos en una base (SQLite, vector DB) y recuperarlos como context adicional. El agente decide qué recordar llamando una tool `remember(fact)` / `recall(query)`.

### Limitaciones de contexto acumulado

- Cada iteración del bucle ReAct añade al menos 2 mensajes (tool_call + tool_result). En 8 iteraciones son 16+ mensajes además del sistema.
- Los tool results largos (salida de `ls -la`, contenido de PDFs) consumen contexto rápido. **Trunca siempre** los resultados de tools.
- Si el contexto excede la ventana, Ollama trunca o falla. Controla el tamaño.

### Estrategias de poda

```python
def trim_messages(messages, max_chars=20000):
    total = sum(len(m.get("content", "")) for m in messages)
    if total <= max_chars:
        return messages
    # Mantener system + últimos N
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    while sum(len(m.get("content", "")) for m in system + rest) > max_chars and len(rest) > 4:
        rest.pop(0)
    return system + rest
```

### Concurrencia

- LangGraph soporta **paralelismo de tools**: si el LLM emite varios tool_calls en una respuesta, pueden ejecutarse en paralelo (asyncio).
- Útil para, por ejemplo, buscar en 3 fuentes a la vez.
- Cuidado con tools que comparten estado (ficheros en escritura): serializa esos o usa locks.

---

## Ejercicio práctico

Construye un agente local que responda preguntas sobre tu propia bóveda de Obsidian. El agente debe:

### Pasos

1. **Definir las tools**:
   - `list_markdown_files(vault_path)`: lista todos los `.md` de la bóveda.
   - `read_file(path)`: lee un fichero.
   - `search_markdown(pattern)`: busca texto dentro de los `.md` con `Path.rglob()` y comparación en Python. Limita la raíz a una copia de prácticas, no a todo tu vault.

2. **Configurar el modelo**: usa `qwen3.5:4b` o un modelo pequeño con tool calling. Comprueba primero que lo tienes:
   ```text
   ollama list
   ollama pull qwen3.5:4b
   ```

3. **Implementar el bucle ReAct** (en Python puro o con LangGraph, a tu elección).

4. **Probar con preguntas reales**:
   - "¿Qué módulos del curso mencionan MLX?"
   - "Resume en 3 bullet points el módulo sobre RAG."
   - "¿Qué comandos de terminal aparecen en el módulo de inferencia?"

5. **Añadir un límite de 6 iteraciones** y medir cuánto tarda. En macOS puedes usar `time uv run python agente.py`; en PowerShell, `Measure-Command { uv run python agente.py }`.

6. **(Opcional) Convertirlo a servidor MCP**: expón las mismas tools vía MCP y conéctalas a Claude Desktop o al inspector MCP.

### Criterios de éxito

- El agente usa **al menos** `list_markdown_files` y `read_file` para responder.
- No entra en bucle infinito: si una tool falla, el agente lo maneja y termina.
- Cumple el objetivo de tiempo que hayas fijado para tu propio equipo.

---

## Recursos

- **ReAct paper**: https://arxiv.org/abs/2210.03629
- **Ollama tool calling docs**: https://docs.ollama.com/capabilities/tool-calling
- **LangGraph docs**: https://langchain-ai.github.io/langgraph/
- **LangChain + Ollama**: https://python.langchain.com/docs/integrations/chat/ollama/
- **MCP spec**: https://modelcontextprotocol.io/
- **MCP Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **MCP servers de referencia**: https://github.com/modelcontextprotocol/servers
- **Biblioteca de modelos con tools**: https://ollama.com/search
- **Anthropic: Building effective agents**: https://www.anthropic.com/research/building-effective-agents

---

> [!tip] Próximos pasos
> Este módulo es base directa del proyecto final del curso ([06-Proyecto-Final](../06-Proyectos/01-Asistente-local-completo.md)), donde integrarás RAG + agente + evaluación. Para evaluar la calidad de las respuestas del agente, consulta [Evaluacion-LLMs-Local](../07-Anexos/A-Evaluacion-local-sin-autoengano.md). Para el stack de inferencia subyacente, repasa [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← 07 - RAG local: pipelines con contexto](01-RAG-local.md) · [Índice](../README.md) · [IA multimodal local: imagen, texto y audio →](03-IA-multimodal-local.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
