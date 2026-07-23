---
title: Bienvenida y método
curso: IA-Local-de-Cero-a-Produccion
modulo: "00.00"
---

# Bienvenida: cómo aprender IA local sin profesor

<!-- CURSO_NAV_TOP -->
[← Índice](../README.md) · [Índice](../README.md) · [Elige una ruta que encaje en tu equipo →](01-Elige-hardware-y-modelo.md)
<!-- /CURSO_NAV_TOP -->



Este curso te acompaña desde "quiero probar un modelo en mi ordenador" hasta "entiendo cómo se sirve para varias personas". No se espera que sepas todo lo que aparece en el índice. Si ya lo supieras, no necesitarías el curso.

## El resultado, antes que las herramientas

Al principio vas a conseguir tres cosas muy concretas:

1. ejecutar un modelo sin enviar la conversación a una API externa;
2. medir cuánto tarda y cuánta memoria usa;
3. explicar por qué has elegido ese modelo y no otro.

Después construirás con él: una API, un RAG, un agente o un sistema de voz. La parte avanzada enseña qué ocurre por debajo y qué cambia cuando pasas de una persona usando el modelo a muchas peticiones concurrentes.

## Qué significa "local"

En este curso, local significa que la inferencia se ejecuta en una máquina que controlas: tu Mac, tu PC con Windows, un servidor propio o una GPU alquilada que administras. El primer caso es el más sencillo y privado. El último se parece más a producción.

```text
tu texto
  → tokenizer
  → modelo cargado en RAM/VRAM
  → cálculo en CPU/GPU
  → tokens de respuesta
```

La primera descarga sí necesita Internet. Después, muchos runtimes pueden funcionar offline si el modelo ya está guardado. Compruébalo de verdad antes de trabajar con información sensible.

## Cómo está diseñado cada capítulo

Cada capítulo empieza por la idea y el problema que resuelve. Después llega una práctica pequeña, una comprobación de la salida y una explicación de cuándo merece la pena usar esa técnica.

No avances por haber llegado al final de la página. Avanza cuando puedas explicar con tus palabras qué ha pasado.

## Tu cuaderno de laboratorio

Crea un Markdown llamado `MI-LABORATORIO.md` fuera del repositorio o dentro de una copia personal. Para cada prueba apunta:

```markdown
## Fecha y experimento

- Equipo:
- Sistema operativo:
- Runtime y versión:
- Modelo y cuantización:
- Contexto:
- Prompt:
- Tokens por segundo:
- Memoria máxima:
- Calidad observada:
- Qué cambiaré en la siguiente prueba:
```

Esta costumbre parece lenta durante cinco minutos y te ahorra horas cuando ya no recuerdas qué combinación iba bien.

## Regla de los 20 minutos

Si un comando falla:

1. lee el error completo, incluida la primera línea;
2. confirma en qué carpeta y entorno estás;
3. ejecuta la comprobación que acompaña al paso;
4. busca el síntoma en [Troubleshooting local](../07-Anexos/E-Troubleshooting-local.md);
5. si llevas 20 minutos sin nueva información, vuelve a la ruta sencilla: Ollama o LM Studio con un modelo pequeño.

Volver a una baseline que funciona no es rendirse. Es depurar con cabeza.

## La prueba de que has aprendido

Al final de cada bloque intenta responder sin mirar:

- ¿Qué entra y qué sale de esta pieza?
- ¿Qué memoria consume?
- ¿Qué cambia si duplico el contexto?
- ¿Cómo sé si ha mejorado?
- ¿Qué dato podría salir de mi equipo?

Si puedes responder con un ejemplo de tu laboratorio, el conocimiento ya es tuyo.

## Antes de instalar nada

Anota:

- sistema operativo y versión;
- RAM total;
- modelo de CPU;
- GPU y VRAM, si es dedicada;
- espacio libre;
- si el equipo es corporativo y tiene restricciones.

En el siguiente capítulo traduciremos esos datos a una ruta realista.

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Índice](../README.md) · [Índice](../README.md) · [Elige una ruta que encaje en tu equipo →](01-Elige-hardware-y-modelo.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
