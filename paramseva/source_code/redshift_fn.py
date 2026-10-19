import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ------------------------ BINNING FUNCTION ------------------------

# binning function
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


