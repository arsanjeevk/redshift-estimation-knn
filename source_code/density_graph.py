import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ecdfs_data = pd.read_csv("data/CDFS.csv")
elais_data = pd.read_csv("data/ELAIS-S1.csv")

z_eCDFS = ecdfs_data['z'].values
z_ELAIS = elais_data['z'].values

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Your redshift data
# --------------------------------------------------
# Replace these with your actual redshift arrays
z_eCDFS = np.asarray(z_eCDFS, dtype=float)
z_ELAIS = np.asarray(z_ELAIS, dtype=float)

# Remove NaN / infinite values
z_eCDFS = z_eCDFS[np.isfinite(z_eCDFS)]
z_ELAIS = z_ELAIS[np.isfinite(z_ELAIS)]

# Keep only physically plotted range
z_eCDFS = z_eCDFS[(z_eCDFS >= 0) & (z_eCDFS <= 6)]
z_ELAIS = z_ELAIS[(z_ELAIS >= 0) & (z_ELAIS <= 6)]

# --------------------------------------------------
# Histogram bins
# --------------------------------------------------
bins = np.arange(0, 6.001, 0.05)

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    z_eCDFS,
    bins=bins,
    density=True,
    alpha=0.55,
    color="steelblue",
    edgecolor="black",
    linewidth=0.4,
    label=f"eCDFS - {len(z_eCDFS)} Sources"
)

ax.hist(
    z_ELAIS,
    bins=bins,
    density=True,
    alpha=0.55,
    color="orange",
    edgecolor="black",
    linewidth=0.4,
    label=f"ELAIS-S1 - {len(z_ELAIS)} Sources"
)

# --------------------------------------------------
# Exact axis limits from reference figure
# --------------------------------------------------
ax.set_xlim(0, 6)
ax.set_ylim(0, 0.16)

# --------------------------------------------------
# Exact tick positions
# --------------------------------------------------
ax.set_xticks(np.arange(0, 6.01, 0.5))
ax.set_yticks(np.arange(0, 0.161, 0.02))

# --------------------------------------------------
# Labels
# --------------------------------------------------
ax.set_xlabel("Redshift (z)")
ax.set_ylabel("Density")

# --------------------------------------------------
# Grid
# --------------------------------------------------
ax.grid(
    True,
    which="major",
    linestyle="-",
    linewidth=0.6,
    alpha=0.35
)

# --------------------------------------------------
# Legend
# --------------------------------------------------
ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=9
)

# --------------------------------------------------
# Formatting
# --------------------------------------------------
ax.tick_params(axis="both", labelsize=9)

plt.tight_layout()
plt.show()