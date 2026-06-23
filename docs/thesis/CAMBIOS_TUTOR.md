# Cambios aplicados para responder al tutor

1. Introducción condensada y formulada mediante preguntas de investigación directas.
2. Corrección de símbolos y comandos LaTeX, incluida `\operatorname{softmax}`.
3. Contextualización específica de CTRW, FBM, LW, ATTM y SBM y de su dificultad para detectar fronteras.
4. Justificación de LSTM, CNN--LSTM, Transformer y ConvTransformer con literatura de AnDi, inferencia puntual y segmentación.
5. Tablas de arquitectura, dropout e hiperparámetros reales: tasa de aprendizaje, lote, épocas, pérdidas, parada, umbral y checkpoint.
6. Comparación explícita de ConvTransformer para `L=100` y `L=200`, sin atribuir causalmente las diferencias a la longitud.
7. Captions ampliados con la orientación de las matrices y las transiciones más problemáticas.
8. Redacción reformulada de manera original y citas situadas junto a las afirmaciones que sustentan.
9. Se añadió un estudio real de robustez de ConvTransformer-v2 con cinco semillas bajo un protocolo reducido y fijo. Se reportan media y desviación estándar, incluidas las convergencias degeneradas observadas.
10. Se incorporó una baseline PELT real sobre las mismas particiones de validación y test. La penalización se eligió sin utilizar el test y sus limitaciones se explican explícitamente.
11. Los resultados complementarios se congelaron como CSV. No se inventaron cifras ni se presentó la comparación con PELT como equivalente al protocolo de AnDi.
12. El tuning exhaustivo, el multi-seed con entrenamiento completo y la reproducción de RANDI o AnDi-ELM permanecen como trabajo futuro.
13. Se incorporó el enlace explícito al repositorio GitHub y se actualizó su entrada bibliográfica.
14. Se recuperó de la versión anterior únicamente un pseudocódigo compacto de generación y una selección de cuatro ejemplos cualitativos de ConvTransformer-v2.
15. ConvTransformer-v3a y v3b se documentan como extensiones exploratorias de validación, sin métricas de test ni afirmaciones de superioridad.
