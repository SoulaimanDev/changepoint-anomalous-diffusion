# Diagnóstico aislado de xLSTM cfg_01, seed 11

Este directorio no pertenece a la campaña oficial. No contiene resultados
seleccionables, checkpoints de campaña ni archivos `run.json` oficiales.
No se ha aplicado el patch numérico ni gradient clipping.

## Ejecución reproducible (PowerShell, desde el workspace)

```powershell
& 'C:/Users/chair/AppData/Local/Programs/Python/Python311/python.exe' -B outputs/xlstm_diagnostic/harness.py `
  --project work/runpod_audit/prepared_project `
  --train 'C:/Users/chair/Downloads/changepoint_new_phase_files/data_synthetic_with_without_changepoint_dx/train_L100_dim1_with_without_dx.h5' `
  --output outputs/xlstm_diagnostic/cpu_cfg01_seed11_NEW `
  --max-seconds 600 --threads 2
```

El directorio de salida debe ser nuevo. La duración es un límite de diagnóstico,
comprobado entre batches; no es un cambio de `max_epochs` del protocolo.
El harness utiliza el factory, loader, normalización, pérdida y Adam originales,
el batch size 256 y `torch.Generator().manual_seed(11)` para la permutación de las
200.000 filas de train. Guarda la permutación completa y su SHA-256.

Lee únicamente el HDF5 train, con comprobación SHA-256 y una guarda de apertura.
No abre archivos de validación, splits NPZ, calibration ni test. Los manifiestos
de texto se leen para verificar la configuración y la procedencia.

La observación usa TorchDispatchMode: ejecuta la operación original y comprueba
su salida sin sustituirla. No cambia pesos, operaciones, RNG ni hiperparámetros.
El test de transparencia verifica forward, gradientes y RNG de un modelo pequeño.
CPU y CUDA no tienen garantía de igualdad bit a bit.

## Límite de fidelidad

El harness se detiene como máximo al final de la primera época. Las siguientes
épocas requieren las decisiones oficiales del controlador sobre validation_tuning,
que no está autorizado en este diagnóstico exclusivamente train. No se omite el
controlador para ejecutar artificialmente épocas adicionales.

Un fallo intermedio detiene inmediatamente la ejecución: si exp produce Inf, el
clamp posterior no se ejecuta. Pérdidas, logits y gradientes aún no calculados
quedan ausentes/null; no se continúa para fabricarlos. Los valores de tensores
se documentan mediante shape, dtype, extremos finitos y conteos NaN/Inf.
`adam_before_step` corresponde al estado previo a la actualización del batch.

Los -Inf de la máscara de localización son intencionales. Se exceptúan únicamente
masked_fill/log_softmax/detach con valores finitos en bins 19..79 y -Inf en todos
los bins restantes; NaN, +Inf y valores no finitos admisibles nunca se exceptúan.

## Ensayo preliminar del monitor

`cpu_cfg01_seed11_001` detectó la máscara intencional durante backward. Es un
falso positivo del monitor inicial, no evidencia de divergencia del modelo.
Se conserva para trazabilidad; no debe usarse como resultado del diagnóstico.
La versión corregida del monitor reconoce ese soporte exacto y tiene tests.

## Tests

```powershell
& 'C:/Users/chair/AppData/Local/Programs/Python/Python311/python.exe' -B -m unittest discover -s outputs/xlstm_diagnostic -p 'test_*.py' -v
```

El test escalar demuestra, sin aplicar ninguna corrección:
`m=-100 -> exp(-m)=Inf -> clamp finito -> gradiente NaN`.
Los otros tests comprueban la máscara intencional, detección de NaN, parada antes
del clamp y ausencia de cambios en forward/backward/RNG por el observador.

La versión final también asocia metadatos de procedencia a los nodos autograd
existentes mediante tracing Python del archivo original. Un test controlado del
MatrixLSTMCore original confirma que el fallo backward se identifica como
ExpBackward0, línea 145, paso temporal 0. No se insertan operaciones de tensor.

`harness_observer_v1.py` conserva exactamente la versión utilizada para los 18
batches de `cpu_cfg01_seed11_002`. `harness.py` añade esa procedencia backward;
`cpu_cfg01_seed11_003` verifica un batch real con esta versión. Los dos JSON
incluyen el hash del harness correspondiente. Véase `diagnostic_report.md`.
