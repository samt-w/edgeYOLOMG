# this script uploads the target dataset to Hugging Face

# uses the HF PAT in .env
import sys
import os
from pathlib import Path
# load access keys from the root .env file
# this relies on a .toml setup allowing the import of the root config.py
try:
    import config
except ImportError:
    print("Error: Could not import config.py. Check .toml path configuration.")
    sys.exit(1)
from huggingface_hub import login, upload_file

# Hugging Face credentials
HF_PAT = os.getenv("HF_PAT")

ARD100_filepath = os.getenv("ZIP_DATA_DIR")

if not HF_PAT or not ARD100_filepath:
    print("Error: Missing HF_PAT or ZIP_DATA_DIR in environment.")
    sys.exit(1)

# login with Hugging Face credentials
login(token = HF_PAT)

file_path = Path(ARD100_filepath)

# upload the dataset file
upload_file(
    path_or_fileobj = file_path,
    path_in_repo = file_path.name,
    repo_id = "stw-hf/ARD100",
    repo_type = "dataset"
)

print("Upload complete.")