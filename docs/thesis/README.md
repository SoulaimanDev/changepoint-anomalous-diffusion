# TFM revisado

Archivos principales:

- `main.tex`: memoria completa revisada.
- `references_revisadas.bib`: bibliografía ampliada y corregida.
- `generate_revised_figures.py`: reconstruye las figuras comparativas a partir de los resultados ya existentes.
- `figures/`: imágenes utilizadas por `main.tex`.
- `notebooks/`: cuaderno ejecutado de ConvTransformer para `L=200`, conservado como evidencia de los resultados incorporados.
- `results/final_tfm_additions/`: CSV congelados del estudio multisemilla de ConvTransformer-v2 y de la baseline PELT.

La memoria contiene únicamente el resumen en español; se ha retirado el `Abstract` por indicación del autor. La Sección 7.4 compara ConvTransformer para `L=100` y `L=200` mediante matrices ordenadas de FNR, nMAE y nRMSE. Los valores de `L=200` proceden de la ejecución global del cuaderno `13_convtransformer_L200_mejorado_profesor_COLAB_GLOBAL_TEST (1).ipynb`.

La revisión para el tutor corrige la expresión de `softmax`, refuerza la justificación bibliográfica de las arquitecturas, amplía los captions de las matrices y documenta los hiperparámetros verificables de cada ejecución. Las diferencias de configuración y la ausencia de tuning homogéneo se presentan explícitamente como limitaciones.

La versión final añade dos experimentos reales. El primero repite ConvTransformer-v2 con cinco semillas sobre un subconjunto fijo de 10.000 trayectorias de entrenamiento y conserva validación y test completos. El segundo evalúa PELT como baseline clásica sobre las mismas particiones. La memoria distingue estos experimentos de la ejecución neuronal principal y no los presenta como una comparación directa con AnDi.

También se añadió el enlace explícito al repositorio
`https://github.com/SoulaimanDev/changepoint-anomalous-diffusion`, un
pseudocódigo compacto de generación y cuatro ejemplos cualitativos de
ConvTransformer-v2. Las variantes ConvTransformer-v3a y v3b se describen
únicamente como extensiones exploratorias evaluadas en validación; no sustituyen
el resultado principal de v2.

En el cuerpo de la memoria, el repositorio se cita mediante la clave
`chair2026repo`; la URL completa se reserva para la bibliografía para evitar
cortes tipográficos en el texto principal.

Compilación recomendada con Tectonic:

```powershell
tectonic -X compile main.tex
```

También puede compilarse con un flujo LaTeX/BibTeX convencional:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Nota científica importante

La red denominada `ConvLSTM` en los cuadernos se presenta como `CNN--LSTM`, porque la implementación aplica capas Conv1D antes de capas LSTM y no utiliza una celda ConvLSTM.

La memoria documenta fielmente las configuraciones y resultados existentes. No afirma que se haya ejecutado un ajuste sistemático de hiperparámetros. El estudio multisemilla corresponde a un régimen reducido y no reemplaza la repetición del entrenamiento completo. El Apéndice A propone el protocolo necesario para completar esa validación sin inventar resultados.
