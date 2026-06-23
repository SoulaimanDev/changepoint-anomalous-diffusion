# Protocolo congelado v2

Este documento fija la versión experimental utilizada como línea base antes de introducir ConvTransformer-v3, repeticiones con varias semillas o una baseline clásica mediante PELT.

## Objetivo

La versión v2 evalúa la detección binaria de puntos de cambio y la localización temporal de la frontera entre dos regímenes de difusión anómala. Los modelos generadores considerados son ATTM, CTRW, FBM, LW y SBM.

## Arquitecturas v2

- LSTM.
- CNN-LSTM, denominada `convlstm` en los cuadernos originales, aunque la implementación corresponde a convoluciones 1D seguidas de LSTM.
- Transformer.
- ConvTransformer-v2.

ConvTransformer-v2 se conserva como la mejor configuración observada en la ejecución principal de `L=100`, no como una arquitectura universalmente superior.

## Datos y particiones

Los resultados congelados corresponden al protocolo binario con trayectorias de longitud `L=100`. El conjunto de prueba incluye trayectorias con cambio y sin cambio. Para las trayectorias positivas, las transiciones ordenadas se definen con filas como modelo del primer segmento y columnas como modelo del segundo segmento.

La versión v2 utiliza incrementos normalizados `dx(t)` como entrada. No se emplean todavía entrada multicanal, etiquetas suaves ni cabezas auxiliares.

## Selección de umbral y checkpoint

Los umbrales binarios se seleccionaron en validación y el conjunto de prueba se reservó para la evaluación final. Los resultados congelados almacenan el umbral finalmente utilizado por cada arquitectura.

## Métricas congeladas

Las métricas guardadas en `results/frozen_v2/` son:

- accuracy;
- precision;
- recall;
- F1;
- FPR;
- FNR;
- MAE y RMSE de localización;
- MAE/RMSE en verdaderos positivos cuando está disponible;
- FPR por generador en trayectorias sin cambio.

## Limitaciones explícitas

Los resultados v2 proceden de una ejecución principal por arquitectura. Por tanto, no permiten estimar la variabilidad debida a inicialización, barajado de datos o estocasticidad del entrenamiento. Las diferencias pequeñas, especialmente entre CNN-LSTM y ConvTransformer-v2, deben interpretarse con prudencia hasta disponer de una comparación multi-semilla.

La comparación ConvTransformer-v2 entre `L=100` y `L=200` se mantiene como resultado descriptivo. No se debe presentar como una ablación causal estricta de la longitud porque el protocolo documentado cambia también otros elementos, como el tamaño de lote, la ponderación de la pérdida y la rejilla de umbrales.

## Regla para nuevos experimentos

Los nuevos experimentos deberán escribirse en carpetas separadas, por ejemplo:

- `results/convtransformer_v3_ablation/`;
- `results/multiseed_v2_vs_v3/`;
- `results/pelt_baseline/`.

La carpeta `results/frozen_v2/` no debe modificarse salvo para corregir errores documentados.

