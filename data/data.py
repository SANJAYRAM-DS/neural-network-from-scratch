import zipfile
import os

# Path to the ZIP file (same folder as this script)
zip_path = os.path.join(os.path.dirname(__file__), "digit-recognizer.zip")

# Create dataset folder path inside data/
extract_path = os.path.join(os.path.dirname(__file__), "dataset")

# Unzip into the 'dataset' folder
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted to:", extract_path)
