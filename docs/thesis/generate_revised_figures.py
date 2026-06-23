from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures" / "revised"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["ATTM", "CTRW", "FBM", "LW", "SBM"]
ARCH = ["LSTM", "CNN-LSTM", "Transformer", "ConvTransformer"]
COLORS = ["#3569A8", "#E68632", "#2D936C", "#C44E52"]


def save(fig, filename):
    fig.savefig(OUT / filename, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


# False-positive rates by homogeneous diffusion model.
fpr_by_model = np.array([
    [0.43345, 0.14955, 0.25645, 0.15610, 0.32670],
    [0.28920, 0.10825, 0.26130, 0.11045, 0.21090],
    [0.25765, 0.08140, 0.21490, 0.07925, 0.20285],
    [0.27505, 0.09145, 0.21995, 0.07930, 0.20045],
])

fig, ax = plt.subplots(figsize=(10.2, 4.8))
x = np.arange(len(MODELS))
width = 0.19
for i, (name, color) in enumerate(zip(ARCH, COLORS)):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, fpr_by_model[i], width, label=name, color=color)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)
ax.set_xticks(x, MODELS)
ax.set_ylim(0, 0.50)
ax.set_ylabel("Tasa de falsos positivos")
ax.set_xlabel("Modelo generador de la trayectoria homogénea")
ax.set_title("Falsos positivos en trayectorias sin punto de cambio")
ax.grid(axis="y", alpha=0.25)
ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
fig.subplots_adjust(bottom=0.25)
save(fig, "fpr_sin_cambio_comparacion.png")


# Global detection metrics.
global_detection = np.array([
    [0.732765, 0.734069, 0.729980, 0.732019],
    [0.789130, 0.797980, 0.774280, 0.785951],
    [0.771165, 0.809284, 0.709540, 0.756137],
    [0.803675, 0.818374, 0.780590, 0.799036],
])
metric_names = ["Exactitud", "Precisión", "Sensibilidad", "F1"]
metric_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
fig, ax = plt.subplots(figsize=(9.8, 4.9))
x = np.arange(len(ARCH))
width = 0.19
for i, (name, color) in enumerate(zip(metric_names, metric_colors)):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, global_detection[:, i], width, label=name, color=color)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
ax.set_xticks(x, ARCH)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Valor de la métrica")
ax.set_title("Comparación global de la detección binaria")
ax.grid(axis="y", alpha=0.25)
ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
fig.subplots_adjust(bottom=0.23)
save(fig, "metricas_globales_deteccion.png")


# Global localization errors.
global_errors = np.array([
    [12.733092, 16.350369],
    [7.664480, 11.918347],
    [8.741436, 12.530686],
    [7.536969, 11.570613],
])
fig, ax = plt.subplots(figsize=(8.8, 4.8))
x = np.arange(len(ARCH))
width = 0.34
bars_mae = ax.bar(x - width / 2, global_errors[:, 0], width, label="MAE", color="#4C78A8")
bars_rmse = ax.bar(x + width / 2, global_errors[:, 1], width, label="RMSE", color="#F58518")
ax.bar_label(bars_mae, fmt="%.2f", padding=3, fontsize=8)
ax.bar_label(bars_rmse, fmt="%.2f", padding=3, fontsize=8)
ax.set_xticks(x, ARCH)
ax.set_ylim(0, 18)
ax.set_ylabel("Error temporal (muestras)")
ax.set_title("Error global de localización del punto de cambio")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, loc="upper right")
save(fig, "errores_globales_localizacion.png")


# Detection trade-off with localization error encoded in the labels.
fpr_global = np.array([0.264450, 0.196020, 0.167210, 0.173240])
recall_global = np.array([0.729980, 0.774280, 0.709540, 0.780590])
mae_global = global_errors[:, 0]
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for i, name in enumerate(ARCH):
    ax.scatter(fpr_global[i], recall_global[i], s=170, color=COLORS[i], edgecolor="white", linewidth=1.2)
    dx = 0.004
    dy = 0.006 if name != "LSTM" else -0.022
    ax.annotate(f"{name}\nMAE={mae_global[i]:.2f}", (fpr_global[i], recall_global[i]),
                xytext=(fpr_global[i] + dx, recall_global[i] + dy), fontsize=8.5)
ax.set_xlim(0.14, 0.29)
ax.set_ylim(0.68, 0.81)
ax.set_xlabel("Tasa de falsos positivos (menor es mejor)")
ax.set_ylabel("Sensibilidad (mayor es mejor)")
ax.set_title("Compromiso entre falsas alarmas, detección y localización")
ax.grid(alpha=0.25)
save(fig, "compromiso_deteccion_localizacion.png")


def matrix(values):
    arr = np.asarray(values, dtype=float)
    np.fill_diagonal(arr, np.nan)
    return arr


fnr = {
    "LSTM": matrix([[0, .28, .42, .20, .39], [.29, 0, .21, .07, .13], [.40, .19, 0, .27, .54], [.14, .05, .22, 0, .21], [.42, .15, .58, .27, 0]]),
    "CNN-LSTM": matrix([[0, .25, .38, .11, .40], [.21, 0, .15, .04, .12], [.36, .15, 0, .17, .48], [.11, .04, .18, 0, .16], [.40, .14, .50, .16, 0]]),
    "Transformer": matrix([[0, .40, .44, .14, .43], [.34, 0, .26, .05, .20], [.43, .26, 0, .24, .55], [.13, .06, .23, 0, .18], [.47, .25, .56, .18, 0]]),
    "ConvTransformer": matrix([[0, .27, .38, .09, .43], [.22, 0, .14, .03, .10], [.36, .14, 0, .16, .50], [.09, .03, .16, 0, .13], [.42, .12, .48, .13, 0]]),
}

mae = {
    "LSTM": matrix([[0, 17.3, 13.7, 12.4, 13.8], [11.1, 0, 9.0, 7.7, 8.7], [15.4, 16.8, 0, 14.0, 14.9], [9.0, 9.1, 9.8, 0, 9.3], [15.8, 17.1, 15.3, 14.5, 0]]),
    "CNN-LSTM": matrix([[0, 8.3, 11.3, 5.3, 11.1], [6.4, 0, 6.3, 3.2, 6.1], [11.3, 6.6, 0, 6.8, 12.6], [5.5, 3.2, 6.5, 0, 5.9], [11.1, 6.4, 12.7, 6.5, 0]]),
    "Transformer": matrix([[0, 9.9, 12.0, 5.8, 11.9], [7.9, 0, 7.8, 3.9, 7.7], [12.1, 8.2, 0, 7.7, 13.4], [6.3, 4.3, 7.8, 0, 7.1], [12.2, 8.1, 13.6, 7.2, 0]]),
    "ConvTransformer": matrix([[0, 8.4, 11.2, 4.9, 11.0], [6.6, 0, 6.6, 2.9, 6.1], [11.1, 6.4, 0, 6.3, 12.3], [5.3, 3.0, 6.2, 0, 5.8], [11.3, 6.4, 12.6, 6.2, 0]]),
}

rmse = {
    "LSTM": matrix([[0, 19.8, 15.7, 14.6, 16.2], [12.5, 0, 10.6, 10.0, 11.0], [17.7, 19.2, 0, 15.9, 17.4], [11.6, 12.5, 11.7, 0, 11.3], [18.8, 19.5, 18.0, 16.6, 0]]),
    "CNN-LSTM": matrix([[0, 10.7, 13.4, 7.7, 13.7], [9.1, 0, 8.7, 5.8, 8.6], [13.7, 9.2, 0, 9.0, 14.8], [8.1, 6.1, 8.5, 0, 8.3], [13.9, 8.9, 14.7, 8.9, 0]]),
    "Transformer": matrix([[0, 11.9, 13.7, 8.0, 14.0], [9.6, 0, 9.2, 6.4, 9.8], [13.7, 9.6, 0, 9.3, 15.2], [8.6, 6.8, 9.4, 0, 9.1], [14.2, 9.7, 15.4, 9.2, 0]]),
    "ConvTransformer": matrix([[0, 10.9, 13.1, 7.5, 13.1], [8.9, 0, 8.9, 5.4, 8.6], [12.9, 8.7, 0, 8.5, 13.9], [7.8, 5.5, 8.3, 0, 8.0], [13.7, 8.9, 14.6, 8.6, 0]]),
}


def heatmap_grid(data, title, cbar_label, filename, vmin, vmax, fmt):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), constrained_layout=True)
    image = None
    for ax, (name, values) in zip(axes.flat, data.items()):
        image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks(range(5), MODELS)
        ax.set_yticks(range(5), MODELS)
        ax.set_xlabel("Segundo modelo")
        ax.set_ylabel("Primer modelo")
        for i in range(5):
            for j in range(5):
                if i == j:
                    text = "--"
                    color = "#555555"
                else:
                    text = format(values[i, j], fmt)
                    color = "white" if values[i, j] < (vmin + vmax) / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)
    fig.suptitle(title)
    cbar = fig.colorbar(image, ax=axes, shrink=0.82, location="right")
    cbar.set_label(cbar_label)
    save(fig, filename)


heatmap_grid(fnr, "Tasa de falsos negativos por transición ordenada", "FNR", "fnr_transiciones.png", 0.0, 0.60, ".2f")
heatmap_grid(mae, "MAE de localización por transición ordenada", "MAE (muestras)", "mae_transiciones.png", 2.5, 18.0, ".1f")
heatmap_grid(rmse, "RMSE de localización por transición ordenada", "RMSE (muestras)", "rmse_transiciones.png", 5.0, 20.0, ".1f")

# Fixed-architecture comparison across trajectory lengths. Localization
# errors are divided by L so that the two panels have the same meaning.
convtransformer_l200_fnr = matrix([
    [0, .4554, .4620, .0180, .2146],
    [.4680, 0, .5504, .0152, .1644],
    [.3666, .4336, 0, .0142, .1512],
    [.0164, .0140, .0158, 0, .0180],
    [.2278, .1742, .2232, .0166, 0],
])
convtransformer_l200_nmae = matrix([
    [0, .102, .102, .007, .055],
    [.096, 0, .110, .006, .045],
    [.085, .095, 0, .005, .042],
    [.007, .006, .005, 0, .008],
    [.055, .049, .049, .008, 0],
])
convtransformer_l200_nrmse = matrix([
    [0, .140, .140, .035, .098],
    [.136, 0, .146, .029, .084],
    [.123, .130, 0, .027, .079],
    [.033, .030, .026, 0, .030],
    [.099, .089, .091, .030, 0],
])


def fixed_architecture_length_figure(length, fnr_values, nmae_values, nrmse_values, filename):
    panels = [
        (fnr_values, "FNR", "Tasa de falsos negativos", 0.0, 0.60, ".2f"),
        (nmae_values, "nMAE = MAE / L", "Error relativo", 0.0, 0.12, ".3f"),
        (nrmse_values, "nRMSE = RMSE / L", "Error relativo", 0.0, 0.15, ".3f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), constrained_layout=True)
    for ax, (values, title, cbar_label, vmin, vmax, fmt) in zip(axes, panels):
        image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(5), MODELS, rotation=35, ha="right")
        ax.set_yticks(range(5), MODELS)
        ax.set_xlabel("Modelo del segundo segmento")
        ax.set_ylabel("Modelo del primer segmento")
        for i in range(5):
            for j in range(5):
                if i == j:
                    text_value = "--"
                    color = "#555555"
                else:
                    text_value = format(values[i, j], fmt)
                    color = "white" if values[i, j] < (vmin + vmax) / 2 else "black"
                ax.text(j, i, text_value, ha="center", va="center", fontsize=8, color=color)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
    fig.suptitle(
        f"ConvTransformer, L={length}: rendimiento por transición ordenada\n"
        "(filas = primer modelo; columnas = segundo modelo)"
    )
    save(fig, filename)


fixed_architecture_length_figure(
    100,
    fnr["ConvTransformer"],
    mae["ConvTransformer"] / 100.0,
    rmse["ConvTransformer"] / 100.0,
    "convtransformer_transiciones_L100.png",
)
fixed_architecture_length_figure(
    200,
    convtransformer_l200_fnr,
    convtransformer_l200_nmae,
    convtransformer_l200_nrmse,
    "convtransformer_transiciones_L200.png",
)

print(f"Figuras guardadas en {OUT}")
