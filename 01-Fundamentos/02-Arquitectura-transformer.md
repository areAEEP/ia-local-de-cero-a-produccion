---
tags:
  - curso/ia-local
  - transformer
  - teoria
curso: IA-Local-de-Cero-a-Produccion
modulo: anexo-arquitectura-transformer
estado: completo
---

# Arquitectura transformer - intuición práctica

<!-- CURSO_NAV_TOP -->
[← Fundamentos: qué es un LLM y qué necesita tu equipo](01-Que-es-un-LLM.md) · [Índice](../README.md) · [Memoria, contexto y KV cache →](03-Memoria-contexto-y-KV-cache-en-Apple-Silicon.md)
<!-- /CURSO_NAV_TOP -->



> [!TIP]
> **Objetivos de aprendizaje**
> - Entender la arquitectura transformer sin perderse en álgebra.
> - Relacionar tokenizer, embeddings, atención, MLP y logits.
> - Conectar esta intuición con [01-Fundamentos](01-Que-es-un-LLM.md), [02-Inferencia](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md) y [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).


## El transformer como tubería de transformación

Un transformer decoder-only convierte una secuencia de tokens en una predicción del siguiente token. No “lee” como una persona; transforma vectores capa a capa.

```text
tokens
  → embeddings
  → bloque transformer 1
  → bloque transformer 2
  → ...
  → bloque transformer N
  → logits del siguiente token
```

Cada bloque mezcla dos operaciones principales:

1. **Atención**: decide qué tokens anteriores importan para el token actual.
2. **MLP**: aplica transformaciones no lineales que almacenan patrones aprendidos.

## Tokenizer: la puerta de entrada

El tokenizer parte texto en unidades. No siempre son palabras completas. Por ejemplo, una palabra rara puede dividirse en varios tokens.

Esto importa porque:

- el coste se mide en tokens, no en palabras;
- idiomas y símbolos distintos pueden tokenizar peor o mejor;
- código, JSON y Markdown pueden consumir muchos tokens;
- extender vocabulario no es trivial.

## Embeddings: tokens como vectores

Cada token se convierte en un vector. Ese vector no es una definición de diccionario; es una posición aprendida en un espacio matemático. El modelo opera sobre esos vectores.

Intuición:

```text
token "Mac" → vector con miles de dimensiones
```

Las capas posteriores transforman ese vector según el contexto.

## Atención: buscar información relevante

La atención produce tres vistas de cada token:

- **Query**: qué necesita este token.
- **Key**: qué ofrece cada token anterior.
- **Value**: qué información se mezcla si se considera relevante.

El modelo compara Query contra Keys y mezcla Values. Esto permite que un token final atienda a elementos lejanos del prompt.

## MLP: memoria distribuida de patrones

Después de la atención, el MLP transforma cada posición. Gran parte del conocimiento práctico del modelo se expresa en estas capas: asociaciones, estilos, patrones de código, formatos y regularidades lingüísticas.

No hay una “neurona de Python” simple. El conocimiento está distribuido.

## Logits y muestreo

Al final, el modelo produce una puntuación por token posible. Esa lista se transforma en probabilidades y el runtime elige el siguiente token.

Parámetros como temperature o top_p no cambian el modelo. Cambian la forma de elegir desde esa distribución.

## Positional encoding: orden en la secuencia

El transformer no lee tokens de izquierda a derecha como un RNN. Procesa todos a la vez. Pero el orden importa: "el perro muerde al hombre" y "el hombre muerde al perro" tienen los mismos tokens en distinto orden.

La *positional encoding* añade información de posición a cada embedding. Sin ella, el modelo trataría tokens en cualquier orden como equivalentes.

```text
embedding del token
  + información de posición
  = vector que sabe qué token es y dónde está
```

## Multi-head attention: múltiples perspectivas

La atención no se calcula una sola vez. Se divide en *heads* paralelos, cada uno enfocando distintos aspectos:

```text
head 1 → relaciones sintácticas
head 2 → referencias largas
head 3 → formato/estructura
head 4 → entidad semántica
...
```

Cada head tiene sus propios Q, K, V. Al final se concatenan y mezclan.

```text
         ┌── head 1 (Q₁, K₁, V₁) ──┐
input ───┤── head 2 (Q₂, K₂, V₂) ──┤── concat ── proyección ── output
         └── head N (Qₙ, Kₙ, Vₙ) ──┘
```

## Layer normalization y conexiones residuales

Cada bloque transformer tiene dos normalizaciones y dos conexiones residuales (skip connections):

```text
x ──→ [Attention] ──→ Add & Norm ──→ [MLP] ──→ Add & Norm ──→ output
 │         ↑                          ↑
 └─────────┘                          │
 └────────────────────────────────────┘
```

Las conexiones residuales permiten que el gradiente fluya durante el entrenamiento y que información cruda se preserve capa a capa. La normalización estabiliza los valores.

## GQA y MQA: optimizaciones modernas

Modelos recientes usan *Grouped-Query Attention* (GQA) o *Multi-Query Attention* (MQA). En vez de tener K y V separados por cada head, comparten K/V entre grupos de heads.

```text
MHA: cada head tiene K, V propios → máxima calidad, máxima memoria
GQA: grupos de heads comparten K, V → equilibrio calidad/memoria
MQA: todos los heads comparten un K, V → mínima memoria, algo menos calidad
```

Esto reduce el tamaño del KV cache, lo que importa mucho en 24 GB. Relacionado: [Memoria-KV-Cache-Apple-Silicon](03-Memoria-contexto-y-KV-cache-en-Apple-Silicon.md).

## Flash attention: optimización de cálculo

Flash attention reorganiza el cálculo de atención para usar mejor la jerarquía de memoria. No cambia los resultados matemáticos; cambia cómo se accede a memoria.

En llama.cpp se activa con `-fa`. En MLX puede estar habilitado por defecto según versión.

## Diagrama completo del flujo

```text
┌─────────────┐
│   tokens     │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│  embeddings +        │
│  positional encoding │
└──────┬───────────────┘
       │
       ▼
┌══════════════════════════════════════════┐
║  Bloque transformer (repetido N veces)    ║
║                                          ║
║  ┌──────────────────────────────────┐    ║
║  │ Multi-Head Attention (con GQA)  │    ║
║  │   Q · Kᵀ → softmax · V          │    ║
║  └────────────┬─────────────────────┘    ║
║      + residual + layer norm             ║
║               │                          ║
║  ┌────────────▼─────────────────────┐    ║
║  │  MLP / Feed-Forward              │    ║
║  │  (dos capas lineales + activación)│    ║
║  └────────────┬─────────────────────┘    ║
║      + residual + layer norm             ║
╚══════════════│═══════════════════════════╝
               │ (repetir N capas)
               ▼
┌─────────────────────┐
│  LM Head (proyección)│
│  → logits por token  │
└──────┬───────────────┘
       │
       ▼
┌─────────────────────┐
│  softmax + muestreo  │
│  (temperature, top_p)│
└──────┬───────────────┘
       │
       ▼
   siguiente token
```

## Relación con cuantización

Cuantizar reduce precisión numérica de los pesos dentro de atención y MLP. Si el error es pequeño, la distribución final apenas cambia. Si el error es grande, cambian los logits y aparecen peores respuestas.

Relacionado: [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md).

## Relación con fine-tuning

Fine-tuning ajusta pesos o adaptadores para cambiar probabilidades. LoRA añade pequeñas rutas entrenables que modifican ciertas transformaciones sin reentrenar todo el modelo.

Relacionado: [04-Fine-Tuning](../04-Adaptar/01-Fine-tuning-con-MLX-en-Mac.md).

## Ejercicio práctico

Explica con tus palabras este flujo:

```text
prompt → tokens → atención/MLP → logits → muestreo → respuesta
```

Después identifica en qué parte actúa cada cosa:

- temperatura;
- cuantización;
- LoRA;
- KV cache;
- tokenizer.

## Recursos

- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Transformer Circuits: https://transformer-circuits.pub/

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Fundamentos: qué es un LLM y qué necesita tu equipo](01-Que-es-un-LLM.md) · [Índice](../README.md) · [Memoria, contexto y KV cache →](03-Memoria-contexto-y-KV-cache-en-Apple-Silicon.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
