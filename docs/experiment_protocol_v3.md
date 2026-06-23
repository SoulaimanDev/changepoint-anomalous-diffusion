# Protocolo experimental ConvTransformer-v3

## Objetivo de la fase v3

La fase ConvTransformer-v3 introduce una extensión metodológica del modelo ConvTransformer-v2 con el objetivo de estudiar si una representación de entrada más rica puede mejorar posteriormente la detección y la localización de puntos de cambio en trayectorias de difusión anómala.

Esta fase está inspirada por la estructura de los problemas de segmentación del Anomalous Diffusion Challenge, donde las trayectorias pueden presentar cambios de régimen dinámico. Sin embargo, ConvTransformer-v3 no se plantea como una reproducción directa de AnDi ni como una afirmación de superioridad frente a sus métodos. La comparación con AnDi solo podría considerarse directa si se utilizara exactamente el mismo protocolo, con las mismas entradas, salidas, métricas y condiciones experimentales.

Los resultados de ConvTransformer-v2 permanecen congelados en `results/frozen_v2/`. Esta nueva fase no modifica dichos resultados.

## Alcance de la variante v3a

La variante ConvTransformer-v3a prepara únicamente el pipeline de entrada multicanal. Esta etapa no produce resultados de entrenamiento, no evalúa rendimiento y no compara v3a con v2. Su objetivo es preparar las características que se utilizarán en una experiencia posterior.

La tarea principal del TFM se mantiene igual que en v2: detección binaria de presencia de punto de cambio y localización temporal cuando dicho cambio existe.

## Entrada multicanal

En ConvTransformer-v3a, la entrada se construye a partir de la secuencia de incrementos de la trayectoria. En lugar de proporcionar solo `dx(t)`, se generan varios canales derivados de la señal observada:

1. `dx_norm`: incrementos estandarizados por trayectoria;
2. `abs_dx`: valor absoluto de los incrementos estandarizados;
3. `dx_squared`: cuadrado de los incrementos estandarizados;
4. `local_variance`: varianza local con una ventana deslizante;
5. `local_abs_mean`: media local del valor absoluto de los incrementos;
6. `lag1_product`: producto `dx_norm(t) dx_norm(t-1)`, usado como descriptor simple de persistencia local.

La entrada multicanal no utiliza etiquetas, ni la posición real del punto de cambio, ni la variable binaria de presencia de cambio. Por tanto, no introduce fuga de información supervisada. Su objetivo es reexpresar los incrementos mediante descriptores locales calculados exclusivamente a partir de la señal observada.

## Control de fuga de información

La construcción de las características multicanal no utiliza:

- la etiqueta `has_cp`;
- la posición real del punto de cambio;
- las etiquetas `modelo1` o `modelo2`;
- las etiquetas de clasificación del modelo generador;
- ninguna información del conjunto de prueba para tomar decisiones de diseño.

Todas las características se calculan directamente a partir de los incrementos observados de cada trayectoria. La longitud temporal se conserva: si la entrada tiene forma `(N, L-1, 1)`, la salida tiene forma `(N, L-1, 6)`.

## Variantes previstas

La fase v3 se organizará en tres variantes progresivas:

- v3a: entrada multicanal;
- v3b: v3a más etiquetas suaves de localización;
- v3c: v3b más cabezas auxiliares para `modelo1` y `modelo2`.

La variante v3a solo prepara la representación multicanal. Las etiquetas suaves de localización y las cabezas auxiliares no forman parte de esta etapa inicial.

## Selección experimental

La selección entre v3a, v3b y v3c se realizará únicamente sobre el conjunto de validación. El conjunto de prueba se reservará para la evaluación final de la variante seleccionada.

| Etapa | Datos utilizados | Decisión tomada | Uso del conjunto de prueba |
|---|---|---|---|
| Ablación v3a/v3b/v3c | Validación | Seleccionar la mejor variante v3 | No |
| Evaluación final de v3 | Prueba | Medir el rendimiento final | Sí, solo evaluación |
| Estudio multisemilla | Prueba con protocolo fijado | Estimar media y desviación estándar | Sí, sin cambiar el protocolo |
| PELT baseline | Validación y prueba | Ajustar penalización en validación y evaluar en prueba | Sí, solo evaluación final |
| Comparación L=100/L=200 | Prueba con protocolo fijado | Analizar el efecto de la longitud si todas las reglas son idénticas | Sí, solo si el protocolo está controlado |

## Relación con AnDi

ConvTransformer-v3 se inspira en la necesidad de capturar información local y temporal relevante en difusión anómala. No obstante, los resultados no se presentarán como una comparación directa con AnDi Task 3 salvo que el protocolo sea idéntico. En este TFM, la tarea principal sigue siendo la detección binaria y la localización temporal del punto de cambio, no la caracterización completa de cada segmento mediante modelo de difusión y exponente anómalo.
