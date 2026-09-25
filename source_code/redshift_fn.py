import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def randomsplit(X, y): # X is the feature matrix and y is the target vector
    

    np.random.seed(42)

    test_indices = np.random.choice(
        len(X),
        round(len(X) * 0.3),
        replace=False
    )

    train_indices = np.array(
        list(set(range(len(X))) - set(test_indices))
    )

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def binDataFunc(redshiftVector, numBins, maxRedshift = 1.5):
    """Function to bin the data

    Args:
        redshiftVector (np.array): 1-d numpy array holding the redshifts to be binned
        numBins (integer): Number of bins to use
        maxRedshift (float, optional): Value to use as the highest bin edge. Defaults to 1.5.

    Returns:
        np.array: List containing the binned redshifts
        np.array: List containing each of the bin edges
        np.array: List containing the centres of the bins
    """
    sortedRedshift = np.sort(redshiftVector, axis=None)
    
    numPerBin = sortedRedshift.shape[0]//numBins #Integer division!
    # Set first bin edge to be the lowest value supplied
    binEdges = [0]
    # Find each of the bin edges
    for i in range(1, numBins):
        binEdges.append(i * numPerBin)
    binEdges.append(sortedRedshift.shape[0]-1)
    # Replace the indices of the bin edges with the bin edge values
    binEdges = sortedRedshift[binEdges]
    binEdges[-1] = maxRedshift
    
    # New list to hold the median of each bins
    newZ = []
    for i in range(1, numBins + 1):
        if i < numBins:
            newZ.append(np.median([binEdges[i-1], binEdges[i]]))
        else:
            newZ.append(np.median(redshiftVector[np.where((redshiftVector >= binEdges[i-1]) & (redshiftVector < np.max(sortedRedshift)))[0]])) 
    # Bin the data
    for i in range(1, numBins + 1):
        if i < numBins:
            if i == 1:
                redshiftVector[np.where((redshiftVector < binEdges[i]))[0]] = newZ[i-1]
            else:
                redshiftVector[np.where((redshiftVector >= binEdges[i-1]) & (redshiftVector < binEdges[i]))[0]] = newZ[i-1]
        else: 
            redshiftVector[np.where((redshiftVector >= binEdges[i-1]))[0]] = newZ[i-1]
    return redshiftVector, binEdges, newZ




# plot classification results

def plot_classification(y_test, final_prediction, bin_edges, bin_values, title="Classification"):
    # Convert true redshift to classes
    y_true_class = np.digitize(y_test, bin_edges[1:-1], right=False)

    # Convert predicted redshift to classes
    y_pred_class = np.argmin(np.abs(final_prediction[:, None] - np.asarray(bin_values)[None, :]), axis=1)

    # Calculate confusion matrix
    num_bins = len(bin_values)
    cm = confusion_matrix(y_true_class, y_pred_class, labels=np.arange(num_bins))

    # Normalize by spectroscopic-redshift bins
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # Transpose: x = spectroscopic redshift, y = estimated redshift
    cm_plot = cm_norm.T

    # Create variable-width confusion matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    mesh = ax.pcolormesh(bin_edges, bin_edges, cm_plot, cmap="viridis", shading="flat", vmin=0, vmax=1)

    # Set bin labels
    bin_values = np.asarray(bin_values)
    ax.set_xticks(bin_values)
    ax.set_yticks(bin_values)

    tick_labels = [f"< {bin_edges[1]:.2f}" if i == 0 else f"> {bin_edges[-2]:.2f}" if i == len(bin_values) - 1 else f"{value:.2f}" for i, value in enumerate(bin_values)]
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)

    # Set labels and title
    ax.set_xlabel("Spectroscopic Redshift")
    ax.set_ylabel("Estimated Redshift")
    ax.set_title(title, loc="left", fontweight="bold")

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Fraction")

    plt.tight_layout()
    plt.show()



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
    ax1.set_ylabel(r"$z_{\rm photo}$")
    ax1.set_xlabel(r"$z_{\rm spec}$"); ax1.set_title(title)

    # Bottom panel: normalized residual
    ax2.scatter(z_spec, delta_z, s=8, color="black", marker=".", alpha=1)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    ax2.axhline(0.15, color="blue", linestyle="--", linewidth=1)
    ax2.axhline(-0.15, color="blue", linestyle="--", linewidth=1)
    ax2.set_xlim(0, zmax)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel(r"$z_{\rm spec}$")
    ax2.set_ylabel(r"$\frac{z_{\rm spec}-z_{\rm photo}}{1+z_{\rm spec}}$")

    plt.tight_layout(pad=0.5)
    plt.show()