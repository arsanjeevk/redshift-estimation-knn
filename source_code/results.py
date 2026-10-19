import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define eta_0.15 scorer
def eta_015(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    delta_z = (y_true - y_pred) / (1 + y_true)

    return np.mean(np.abs(delta_z) > 0.15)

# ------------------------ REGRESSION RESULTS ------------------------

# calculate regression results
def regression_results(y_test, y_pred, test_name="Regression Test"):
    y_test = np.asarray(y_test, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    delta_z = (y_test - y_pred) / (1 + y_test)

    eta_015_value = eta_015(y_test, y_pred) * 100

    sigma = np.std(delta_z)
    eta_2sigma_value = np.mean(np.abs(delta_z) > 2 * sigma) * 100

    median_delta_z = np.median(delta_z)
    mad = np.median(np.abs(delta_z - median_delta_z))
    sigma_nmad = 1.4826 * mad

    mse = mean_squared_error(y_test, y_pred)

    results = {
        "eta_0.15": eta_015_value,
        "eta_2sigma": eta_2sigma_value,
        "sigma": sigma,
        "sigma_NMAD": sigma_nmad,
        "R^2": r2_score(y_test, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_test, y_pred),
    }

    print(f"\n{test_name} Results:")
    # for metric, value in results.items():
    #     print(f"{metric}: {value:.4f}")

    return {key: float(value) for key, value in results.items()}


# Plot regression scatter graphs
def plot_regression_scatter(y_test, y_pred, title="Regression test name"):
    z_spec = np.asarray(y_test)
    z_phot = np.asarray(y_pred)

    # Normalized residual
    delta_z = (z_spec - z_phot) / (1 + z_spec)

    # Metricsevaluated on 393 test objects.


    sigma = np.std(delta_z)
    median_delta_z = np.median(delta_z)
    mad = np.median(np.abs(delta_z - median_delta_z))
    sigma_nmad = 1.4826 * mad
    eta_015 = np.mean(np.abs(delta_z) > 0.15) * 100

    # Plot range
    z_max = max(np.max(z_spec), np.max(z_phot))
    z_line = np.linspace(0, z_max, 500)

    # ±0.15(1 + zspec) boundaries
    upper = z_line + 0.15 * (1 + z_line)
    lower = z_line - 0.15 * (1 + z_line)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(6, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top panel
    ax1.scatter(z_spec, z_phot, s=2, color="black")

    ax1.plot(z_line, z_line, "r--", linewidth=1)
    ax1.plot(z_line, upper, "b--", linewidth=1)
    ax1.plot(z_line, lower, "b--", linewidth=1)

    ax1.set_ylabel(r"$z_{\rm photo}$", fontsize=18)
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 4.6)
    ax1.set_title(title, fontsize=15)

    ax1.tick_params(axis="both", labelsize=11)

    ax1.text(
        0.08, 0.88,
        rf"$N={len(z_spec)}$" "\n"
        rf"$\sigma={sigma:.2f}$" "\n"
        rf"$NMAD={sigma_nmad:.2f}$" "\n"
        rf"$\eta={eta_015:.1f}\%$",
        transform=ax1.transAxes,
        fontsize=12,
        va="top"
    )

    # Bottom panel
    ax2.scatter(z_spec, delta_z, s=2, color="black")

    ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    ax2.axhline(0.15, color="blue", linestyle="--", linewidth=1)
    ax2.axhline(-0.15, color="blue", linestyle="--", linewidth=1)

    ax2.set_xlabel(r"$z_{\rm spec}$", fontsize=18)
    ax2.set_ylabel(
        r"$\frac{z_{\rm spec}-z_{\rm photo}}{1+z_{\rm spec}}$",
        fontsize=20
    )
    ax2.set_xlim(0, 4)
    ax2.set_ylim(-0.5, 0.5)

    ax2.tick_params(axis="both", labelsize=11)

    plt.tight_layout()

    # Save plot using the function title as the filename
    output_dir = "/home/sk/Projects/Astrophysics/output/regression"
    os.makedirs(output_dir, exist_ok=True)
    filename = title.replace(" ", "_") + ".png"
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()



# ------------------------- CLASSIFICATION RESULTS -------------------------

# Calculate classification results
def classification_results(y_test, y_pred, test_name="Classification Test"):
    y_test = np.asarray(y_test, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    # Classification accuracy
    # accuracy = np.mean(y_test == y_pred)

    # Normalized residuals
    delta_z = (y_test - y_pred) / (1 + y_test)

    # eta_0.15
    eta_015_value = np.mean(np.abs(delta_z) > 0.15) * 100

    # sigma
    sigma = np.std(delta_z)

    # eta_2sigma
    eta_2sigma = np.mean(np.abs(delta_z) > 2 * sigma) * 100

    # sigma_NMAD
    sigma_nmad = 1.48 * np.median(
        np.abs(delta_z - np.median(delta_z))
    )

    results = {
        # "Accuracy": accuracy,
        "eta_0.15": eta_015_value,
        "eta_2sigma": eta_2sigma,
        "sigma": sigma,
        "sigma_NMAD": sigma_nmad
    }

    print(f"\n{test_name} Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return {key: float(value) for key, value in results.items()}



# Plot classification results with continuous redshift scatter plot

def plot_classification_scatter(y_test, final_prediction, title="Classification Results"):
    z_spec = np.asarray(y_test, dtype=float)
    z_photo = np.asarray(final_prediction, dtype=float)

    delta_z = (z_spec - z_photo) / (1 + z_spec)
    zmax = np.ceil(max(np.max(z_spec), np.max(z_photo)))
    z = np.linspace(0, zmax, 500)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

    # Top panel: z_spec vs z_photo
    ax1.scatter(z_spec, z_photo, s=8, color="black", marker=".", alpha=1)
    ax1.plot(z, z, "r--", linewidth=1)
    ax1.plot(z, z + 0.15 * (1 + z), "b--", linewidth=1)
    ax1.plot(z, z - 0.15 * (1 + z), "b--", linewidth=1)
    ax1.set_xlim(0, zmax)
    ax1.set_ylim(0, zmax)
    ax1.set_ylabel(r"$z_{\rm photo}$", fontsize=18)
    ax1.set_title(title)

    # Bottom panel: normalized residual
    ax2.scatter(z_spec, delta_z, s=8, color="black", marker=".", alpha=1)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    ax2.axhline(0.15, color="blue", linestyle="--", linewidth=1)
    ax2.axhline(-0.15, color="blue", linestyle="--", linewidth=1)
    ax2.set_xlim(0, zmax)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel(r"$z_{\rm spec}$", fontsize=18)
    ax2.set_ylabel(r"$\frac{z_{\rm spec}-z_{\rm photo}}{1+z_{\rm spec}}$", fontsize=20)

    plt.tight_layout(pad=0.5)

    # Save plot
    output_dir = "/home/sk/Projects/Astrophysics/output/classification"
    os.makedirs(output_dir, exist_ok=True)
    filename = title.replace(" ", "_") + ".png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")

    plt.show()