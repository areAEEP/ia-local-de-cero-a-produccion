---
title: Tu primera IA local
curso: IA-Local-de-Cero-a-Produccion
modulo: "00.03"
---

# Tu primera sesión de IA local

<!-- CURSO_NAV_TOP -->
[← Prepara Windows o macOS](02-Instalacion-Windows-y-macOS.md) · [Índice](../README.md) · [Elige tu itinerario y comprueba el aprendizaje →](04-Rutas-y-autoevaluacion.md)
<!-- /CURSO_NAV_TOP -->



Aquí buscamos una victoria pequeña y verificable: descargar un modelo, conversar con él, llamar a su API y medir una respuesta.

## 1. Elige un modelo pequeño

Abre la [biblioteca de Ollama](https://ollama.com/search) y comprueba que el tag existe. Para la primera sesión:

- 8 GB: `qwen3.5:0.8b` o un modelo equivalente de 1B;
- 16 GB o más: `qwen3.5:4b`;
- si prefieres otra familia: busca `gemma3` y elige 1B o 4B.

Los modelos y tags cambian. Si uno ya no existe, no es un examen trampa: elige otro del mismo tamaño y anótalo.

```bash
ollama pull qwen3.5:0.8b
ollama run qwen3.5:0.8b
```

PowerShell usa los mismos comandos:

```powershell
ollama pull qwen3.5:0.8b
ollama run qwen3.5:0.8b
```

Dentro del chat, prueba:

```text
Explícame qué es un modelo de lenguaje en tres frases.
Después, dame un ejemplo y reconoce una limitación de tu explicación.
```

Sal con `/bye`.

## 2. Asegúrate de que es local

```text
ollama ps
ollama list
```

Desconecta el Wi-Fi durante una pregunta breve. Si el modelo ya está descargado y Ollama no tiene activada ninguna función cloud, debe seguir respondiendo. Vuelve a conectar la red al terminar.

> [!note] Una comprobación honesta
> Que responda offline prueba dónde se ejecuta la inferencia. No prueba por sí solo que ninguna interfaz, plugin o herramienta adicional envíe datos cuando la red está activa.

## 3. Llama a la API

### macOS

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:0.8b",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Resume en una frase qué significa inferencia local."}
    ]
  }'
```

### Windows PowerShell

```powershell
$body = @{
  model = "qwen3.5:0.8b"
  stream = $false
  messages = @(
    @{ role = "user"; content = "Resume en una frase qué significa inferencia local." }
  )
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Uri "http://localhost:11434/api/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

La interfaz de terminal y la API usan el mismo modelo. Ha cambiado el cliente, no la inteligencia.

## 4. Repite desde Python

Crea `primera_llamada.py` en tu carpeta de laboratorio:

```python
import requests

payload = {
    "model": "qwen3.5:0.8b",
    "stream": False,
    "messages": [
        {
            "role": "user",
            "content": "Dame dos ventajas y dos límites de ejecutar IA en local.",
        }
    ],
}

response = requests.post(
    "http://localhost:11434/api/chat",
    json=payload,
    timeout=120,
)
response.raise_for_status()
data = response.json()

print(data["message"]["content"])
print("Tokens de salida:", data.get("eval_count"))
print("Duración de generación (ns):", data.get("eval_duration"))
```

Ejecuta en ambos sistemas:

```text
uv run python primera_llamada.py
```

Calcula tokens por segundo:

```python
tokens = data.get("eval_count", 0)
seconds = data.get("eval_duration", 0) / 1_000_000_000
print("Tokens/s:", round(tokens / seconds, 2) if seconds else "sin dato")
```

## 5. Tu primer experimento

Haz tres ejecuciones con el mismo prompt:

1. el modelo pequeño;
2. otro modelo de una familia distinta;
3. el primero con una respuesta más larga.

Apunta:

| Modelo | Tamaño en disco | Tokens/s | ¿Sigue bien la instrucción? | Error factual |
|---|---:|---:|---|---|
| | | | | |

No decidas por una sola respuesta. Usa al menos cinco prompts: resumen, extracción, español, razonamiento sencillo y una pregunta cuya respuesta puedas comprobar.

## 6. Limpieza consciente

Lista modelos:

```text
ollama list
```

Elimina solo el que ya no quieras:

```text
ollama rm NOMBRE:TAG
```

No borres la carpeta completa de modelos. Practicar con comandos concretos evita sustos.

## Criterio de salida

Has terminado cuando puedas explicar:

- qué parte hizo Ollama;
- qué parte corresponde al modelo;
- por qué dos tags ocupan distinto;
- dónde viste evidencia de velocidad;
- qué limitación observaste sin que te la dijera el propio modelo.

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Prepara Windows o macOS](02-Instalacion-Windows-y-macOS.md) · [Índice](../README.md) · [Elige tu itinerario y comprueba el aprendizaje →](04-Rutas-y-autoevaluacion.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
