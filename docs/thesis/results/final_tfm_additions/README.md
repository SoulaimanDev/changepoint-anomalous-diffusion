# Resultados adicionales de la versión final

Los CSV de esta carpeta congelan los resultados utilizados en las nuevas
secciones de la memoria:

- estudio multisemilla de ConvTransformer-v2 con semillas 1, 2, 3, 42 y 123;
- baseline PELT sobre el test completo;
- comparación entre la ejecución principal de ConvTransformer-v2 y PELT.

El estudio multisemilla utiliza un subconjunto fijo de 10.000 trayectorias de
entrenamiento, validación completa y test completo. No sustituye la ejecución
principal de ConvTransformer-v2.

La penalización 8 de PELT se eligió previamente mediante validación y se aplicó
después al test completo. El test no se utilizó para seleccionar la
penalización.
