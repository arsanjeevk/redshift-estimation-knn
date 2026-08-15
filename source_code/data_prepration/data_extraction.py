from astropy.io import fits
import pandas as pd

# Path to the FITS file
file_path = "/home/sk/Projects/Astrophysics/data/ATLAS_complete_DR2.fits"

# Open FITS file
with fits.open(file_path) as hdul:
    # Read the first table extension
    data = hdul[1].data

# Convert to Pandas DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv("data/ALTAS.csv", index=False)
print("CSV file saved:", "data/ALTAS.csv")

# Read the CSV file
df = pd.read_csv("data/ALTAS.csv")

# Remove leading/trailing whitespace from field names
df["field"] = df["field"].str.strip()

# Split the datasets
df_cdfs = df[df["field"] == "CDFS"].copy()
df_elais_s1 = df[df["field"] == "ELAIS-S1"].copy()

# Check
print("CDFS shape:", df_cdfs.shape)
print("ELAIS-S1 shape:", df_elais_s1.shape)
print("\nField distribution:")
print(df["field"].value_counts())

# Save the split datasets to CSV files
df_cdfs.to_csv("data/CDFS.csv", index=False)
df_elais_s1.to_csv("data/ELAIS-S1.csv", index=False)