---
tags:
  - curso/ia-local
  - memoria
  - kv-cache
  - hardware
curso: IA-Local-de-Cero-a-Produccion
modulo: anexo-memoria-kv-cache
estado: completo
---

# Memoria, contexto y KV cache

<!-- CURSO_NAV_TOP -->
[← Arquitectura transformer - intuición práctica](02-Arquitectura-transformer.md) · [Índice](../README.md) · [02 - Inferencia local →](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md)
<!-- /CURSO_NAV_TOP -->



> [!NOTE]
> **Linux, Windows y macOS**
> La fórmula de la KV cache no cambia. En Apple Silicon todo compite por memoria unificada; en Linux o Windows con GPU dedicada debes vigilar VRAM y RAM por separado. Consulta [Plataformas y comandos](../PLATAFORMAS-Y-COMANDOS.md).


> [!TIP]
> **Objetivos de aprendizaje**
> - Entender por qué la memoria limita la IA local.
> - Separar memoria de pesos, contexto, KV cache y sistema.
> - Relacionar RAM, VRAM y memoria unificada con la inferencia práctica.


## La memoria no es un número único

Cuando dices “tengo 24 GB”, primero aclara de qué memoria hablas. En un Mac suele ser memoria unificada. En un PC puede ser RAM del sistema y una cantidad distinta de VRAM. El presupuesto se reparte entre:

- sistema operativo y procesos abiertos;
- pesos del modelo;
- KV cache;
- buffers temporales;
- servidor o runtime;
- navegador, IDE, Docker, etc.

Por eso un modelo que “teóricamente cabe” puede ir mal si el sistema no tiene margen.

## Pesos del modelo

Los pesos son el coste fijo principal. Dependen de parámetros y precisión.

```text
7B FP16 ≈ 14 GB solo pesos
7B Q4 ≈ 4-5 GB solo pesos
```

La cuantización de [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md) reduce este coste fijo.

## KV cache

Durante generación, el modelo guarda keys y values de tokens anteriores para no recalcular todo el contexto. Esto acelera, pero consume memoria proporcional al contexto.

```text
más contexto
  → más entradas en KV cache
  → más memoria
  → posible menor velocidad
```

Por eso subir de 4k a 32k no es gratis, aunque el modelo sea el mismo.

## Memoria unificada en Apple Silicon

En Apple Silicon, CPU y GPU comparten memoria. Ventaja: no hay una VRAM separada pequeña que limite. Inconveniente: todo compite por el mismo recurso.

```text
CPU + GPU + sistema + apps → misma memoria física
```

Si macOS entra en presión de memoria, la experiencia cae. Puede haber swapping, cierres o latencias enormes.

## VRAM dedicada en Windows

Con una GPU dedicada, los pesos y la KV cache rinden mejor si permanecen en VRAM. Cuando no caben, llama.cpp puede descargar parte de las capas a RAM/CPU, pero el intercambio a través de PCIe suele reducir mucho la velocidad.

```text
GPU dedicada → VRAM rápida para pesos y KV cache
CPU → RAM más grande, normalmente más lenta para inferencia
```

Vigila ambos presupuestos con `nvidia-smi` y el Administrador de tareas. Un error OOM de CUDA puede aparecer aunque todavía sobre RAM del sistema.

## Ancho de banda: el cuello oculto

La inferencia de LLMs mueve muchos pesos desde memoria. Aunque haya cómputo suficiente, el ancho de banda puede limitar tokens/segundo. En un M2, el ancho de banda ronda el orden de 100 GB/s, muy inferior a GPUs dedicadas grandes.

Implicación: un modelo más pequeño y bien cuantizado puede sentirse mejor que uno grande que apenas cabe.

## Fórmula mental

```text
uso_total ≈ pesos_cuantizados + KV_cache + buffers + macOS/apps
```

Regla de curso:

```text
Q4 ≈ parámetros_B × 0.6 GB
+ 2-4 GB contexto razonable
+ margen para sistema, runtime y aplicaciones
```

No es exacta, pero evita errores gruesos.

## Cálculo real del KV cache

La fórmula del tamaño del KV cache es:

```text
KV_cache = 2 × n_layers × n_heads_kv × d_head × context_length × bytes_per_element
```

Desglose:

- `2`: key y value (dos tensores por capa).
- `n_layers`: número de bloques transformer.
- `n_heads_kv`: número de heads de K/V (reducido si hay GQA/MQA).
- `d_head`: dimensión por head.
- `context_length`: tokens de contexto activos.
- `bytes_per_element`: 2 (FP16) o 1 (INT8) o 0.5 (INT4).

### Ejemplo numérico: Qwen2.5 7B

Valores aproximados para un modelo 7B típico:

```text
n_layers      = 28
n_heads_kv    = 4 (con GQA; sin GQA sería 28)
d_head        = 128
context       = 4096
dtype         = FP16 (2 bytes)
```

Sin GQA (MHA clásica):

```text
KV = 2 × 28 × 28 × 128 × 4096 × 2 = 1.6 GB
```

Con GQA (4 heads kv):

```text
KV = 2 × 28 × 4 × 128 × 4096 × 2 = 0.23 GB
```

La diferencia es enorme. GQA reduce el KV cache ~7x en este caso.

Si subes el contexto a 32k con GQA en FP16:

```text
KV = 2 × 28 × 4 × 128 × 32768 × 2 = 1.9 GB
```

Sin GQA en 32k:

```text
KV = 2 × 28 × 28 × 128 × 32768 × 2 = 13.0 GB
```

Esto explica por qué modelos con GQA son mucho más prácticos para contexto largo en 24 GB.

### Cuantización del KV cache

Si cuantizas K y V a INT8 (1 byte) en el ejemplo con GQA y 4k de contexto:

```text
KV = 2 × 28 × 4 × 128 × 4096 × 1 = 0.12 GB
```

A INT4 (0.5 bytes):

```text
KV = 2 × 28 × 4 × 128 × 4096 × 0.5 = 0.06 GB
```

El ahorro es real, pero puede introducir inestabilidad. Mide calidad antes de aceptarlo.

## Señales de que te falta memoria

- El equipo entero se vuelve lento al generar.
- Activity Monitor, el Administrador de tareas o `nvidia-smi` muestran presión alta.
- El tiempo hasta primer token crece mucho.
- El modelo falla al cargar.
- Funciona con contexto 4k, pero no con 16k.

## Decisiones prácticas

Si te falta memoria, en este orden:

1. cierra apps pesadas;
2. baja contexto;
3. usa Q4 en vez de Q5/Q6;
4. baja de 14B a 7B;
5. prueba el backend adecuado: MLX/Metal en Mac; CUDA o Vulkan en Windows;
6. manda el caso a cloud si necesitas tamaño mayor.

## Ejercicio práctico

Ejecuta un mismo modelo con contexto 2048, 4096 y 8192. Observa memoria y latencia. Escribe qué parte del comportamiento cambia y cuál no.

## Recursos

- Apple Silicon overview: https://developer.apple.com/metal/
- llama.cpp: https://github.com/ggml-org/llama.cpp
- MLX: https://github.com/ml-explore/mlx
- [03-Cuantizacion](../02-Uso-local/02-Cuantizacion-y-formatos.md)

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---


Curso creado por [@are_agi](https://twitter.com/are_agi).

---

<!-- CURSO_NAV_BOTTOM -->
[← Arquitectura transformer - intuición práctica](02-Arquitectura-transformer.md) · [Índice](../README.md) · [02 - Inferencia local →](../02-Uso-local/01-Inferencia-con-Ollama-llama.cpp-y-MLX.md)
<!-- /CURSO_NAV_BOTTOM -->

Curso creado por [@are_agi](https://twitter.com/are_agi).
