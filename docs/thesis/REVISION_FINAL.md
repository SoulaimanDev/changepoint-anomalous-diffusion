# Revisión final de la memoria

## Elementos añadidos

- Enlace explícito y referencia bibliográfica del repositorio GitHub.
- Pseudocódigo compacto de generación sintética.
- Cuatro ejemplos cualitativos ya existentes de ConvTransformer-v2.
- Discusión exploratoria de ConvTransformer-v3a y ConvTransformer-v3b.
- Documentación del estudio multisemilla y de la baseline PELT.

## Elementos recuperados de la versión anterior

Solo se recuperaron dos componentes con valor metodológico:

1. la idea del pseudocódigo de generación, reescrita y actualizada;
2. una selección de cuatro predicciones cualitativas de ConvTransformer-v2.

## Elementos no recuperados

- introducción extensa y circular;
- secciones de transición del tipo «relación con las secciones siguientes»;
- explicaciones generales ya cubiertas de forma más precisa;
- figuras cualitativas redundantes de las cuatro arquitecturas;
- uso de «ConvLSTM» para la arquitectura Conv1D seguida de LSTM;
- afirmaciones de superioridad no respaldadas por un protocolo común;
- mezcla de resultados completos con ensayos rápidos de validación.

## Alcance de las variantes v3

ConvTransformer-v3a y v3b se presentan como experimentos preliminares de
validación. No existen métricas de test para estas variantes y no se consideran
sustitutas de ConvTransformer-v2.

## Reproducibilidad

La memoria presenta la reproducibilidad como una práctica metodológica basada
en conservar conjuntamente el código, los cuadernos, las configuraciones, las
semillas, las versiones de las dependencias, los resultados agregados y el
commit utilizado.
