# Diagnóstico reproducible: xLSTM cfg_01, seed 11

## Resultado

**INCONCLUSO: el fallo real no se ha reproducido en el alcance ejecutado.**

- A) No se reprodujo Nonfinite gradient con los datos reales dentro de la ventana CPU examinada.
- B) Se completaron los batches 1..18 de la época 1; la época completa NO se
  terminó. Índices del código: epoch=0, batches=0..17.
  No existe epoch/batch de fallo observado.
- C) Ningún tensor inesperadamente no finito detectado.
- D) Ninguna operación responsable de un fallo real identificada.
- E) xlstm.py:145 **ni confirmada ni descartada como causa real de RunPod**.

La ejecución fue limitada a 600 segundos, comprobados entre batches: finalizó
tras aproximadamente 607,7 s y 18 actualizaciones Adam. Se procesaron 4.608 de las
200.000 trayectorias de la primera época (2,304 %), con batch size 256. No se
interpreta este resultado como estabilidad de una época completa ni de la campaña.

## Reproducibilidad y límites

Modelo original de 173.266 parámetros, cfg_01 original, seed 11, PyTorch
2.11.0+cpu, NumPy 1.26.4, dos threads CPU. Se utilizaron el loader y
normalización oficiales sobre las 200.000 filas de train; no se redujo el tamaño
del conjunto antes de generar la permutación oficial con su generador seed 11.

SHA-256 de la permutación int64 little-endian:
`b6bb44262059e1ec2e4fabe1cd2831340e25c2efc92cedb1e9f46ac7ae777fb0`.

Una repetición del primer batch con la versión final del harness completó también
backward y Adam. Sus pérdidas, resúmenes de logits, índices y permutación coinciden
exactamente con los registrados en el primer ensayo. Esto no demuestra igualdad
bit a bit con CUDA ni reproducción de la trayectoria remota completa.

No se ejecuta una segunda época: el protocolo requiere decisiones sobre
validation_tuning al cerrar la primera y esta autorización es exclusivamente train.

## Evidencia numérica observada

Último batch completo (epoch 0, batch 17):

| Tensor | Valor/rango finito |
|---|---|
| loss total | 5.5235676765441895 |
| detection_loss | 0.893191397190094 |
| localization_loss | 2.0390405654907227 |
| m_state, segundo bloque, paso 98 | [-1.1024963855743408, 224.9728240966797] |
| exp(-m_state), mismo punto | [0.0, 3.0116748809814453] |
| después del clamp, mismo punto | [0.0, 3.0116748809814453] |

El cero es compatible con underflow de la exponencial para m_state positivo
grande; es finito. No constituye el overflow para m_state muy negativo demostrado
por el test escalar, ni prueba por sí solo una causa del fallo remoto.

Se observaron outputs de operaciones de forward, pérdida, backward y Adam.
Los JSON registran shapes, dtypes, extremos finitos, conteos NaN/Inf, logits,
pérdidas, último estabilizador, estado Adam previo al step, índices exactos y
gradientes no finitos presentes. En un fallo durante forward, los valores aún no
calculados quedan null; no se continúa después del fallo para obtenerlos.

## Defecto escalar y tests

Seis tests pasan. El test escalar original establece:

```text
m_state = -100 (float32)
exp(-m_state) = Inf
clamp(max=1e30) = finito
backward respecto a m_state = NaN
```

Otro test usa MatrixLSTMCore original con gates controlados exclusivamente en
memoria: forward finito, fallo backward identificado en ExpBackward0, línea 145,
paso 0. Es un caso construido para verificar la instrumentación, NO cfg_01 real.
No se ha aplicado ninguna corrección numérica.

## Artefactos y ensayo preliminar

- `harness.py`: versión final con procedencia backward.
- `harness_observer_v1.py`: versión exacta que ejecutó los 18 batches.
- `test_scalar.py`, `test_observer.py`, `tests.log`: seis tests y sus resultados.
- `cpu_cfg01_seed11_002/diagnostic.json`: diagnóstico principal, 18 batches.
- `cpu_cfg01_seed11_003/diagnostic.json`: repetición del primer batch con versión final.
- Cada ejecución guarda `epoch_000_order.npy` y hash del harness.
- `summary.json`: conclusión e integridad en formato estructurado.
- `cpu_cfg01_seed11_001`: ensayo preliminar del monitor, conservado pero INVALIDADO
  como evidencia de divergencia. Detectó los -Inf intencionales de la máscara de
  localización. La excepción del monitor se limitó al soporte exacto de esa máscara
  y se verificó mediante tests; no se modificó la pérdida oficial.

## Integridad

Código sellado verificado antes y después:
`175c5ba43e7aa5d2836aa1daf497045af4163b1178529924976985f5e2f0ea65`.

ZIP original verificado contra su manifiesto:
`003e84d8140ed932022745b51153e1f177ee6e7c7812f3c78680a90df7bf9cd2`.

Train SHA-256:
`366f94e0962a164bcf63d235ea26540777fae7958014de33c54de7f98c9e5d35`.

No se escribieron archivos de campaña, registries, checkpoints ni attempts
oficiales. No se accedió a validation_tuning, validation_calibration, splits NPZ ni
test. No hubo GPU, clipping ni cambios de hiperparámetros o modelo.

El siguiente paso diagnóstico, sujeto a nueva aprobación, sería ampliar el tiempo
CPU dentro de la primera época y/o contrastar el epoch/batch de los registros
remotos. No corresponde aprobar un patch como solución del fallo real basándose
únicamente en el caso escalar.
