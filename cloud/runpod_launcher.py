import os
import sys
import runpod
import base64

### ------- CONFIG -------------------------------------------------
# load access keys from the root .env file
# this relies on a .toml setup allowing the import of the root config.py
try:
    import config
except ImportError:
    print("Error: Could not import config.py. Check .toml path configuration.")
    sys.exit(1)

# GitHub credentials
GITHUB_PAT = os.getenv("GITHUB_PAT")

# RunPod credentials
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
NETWORK_VOLUME_ID = os.getenv("RUNPOD_NETWORK_VOLUME_ID")

# ClearML credentials
CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST")
CLEARML_API_HOST = os.getenv("CLEARML_API_HOST")
CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST")
CLEARML_ACCESS = os.getenv("CLEARML_API_ACCESS_KEY")
CLEARML_SECRET = os.getenv("CLEARML_API_SECRET_KEY")

if not all([GITHUB_PAT,
            RUNPOD_API_KEY,
            NETWORK_VOLUME_ID,
            CLEARML_WEB_HOST,
            CLEARML_API_HOST,
            CLEARML_FILES_HOST,
            CLEARML_ACCESS,
            CLEARML_SECRET]):
    print("Error: Missing one or more required environment variables in .env.")
    sys.exit(1)

# initialise RunPod SDK
runpod.api_key = RUNPOD_API_KEY

# hardware setup
GPU_ID = "NVIDIA GeForce RTX 4090"
CONTAINER_DISK_GB = 600

# set up environment variables as a dictionary so they can be securely
# sent to RunPod container
pod_env_vars = {
    "GITHUB_PAT": GITHUB_PAT,
    "RUNPOD_API_KEY": RUNPOD_API_KEY,
    "CLEARML_WEB_HOST": CLEARML_WEB_HOST,
    "CLEARML_API_HOST": CLEARML_API_HOST,
    "CLEARML_FILES_HOST": CLEARML_FILES_HOST,
    "CLEARML_API_ACCESS_KEY": CLEARML_ACCESS,
    "CLEARML_API_SECRET_KEY": CLEARML_SECRET
}
### ----------------------------------------------------------------

### ------- REMOTE EXECUTION SCRIPT -------------------------------------------------
# this bash script executes inside the RunPod container upon boot
# 'set -e' triggers pod termination if any command fails
pod_bash_script = """
set -e

# define cleanup function to terminate pods correctly, with retry logic
function cleanup {
    local exit_code=$?
    echo "Triggering Pod termination (exit code $exit_code)..."
    for i in 1 2 3; do
        # -s: silent, -S: show error, -f: fail on HTTP errors (allows && break to work)
        curl -s -S -f -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \\
             -H "Authorization: Bearer ${RUNPOD_API_KEY}" && break
        echo "Termination attempt $i failed, retrying in 5 seconds..."
        sleep 5
    done
}

# Trap EXIT runs the cleanup function even if script crashes
trap cleanup EXIT

echo "extracting pre-processed dataset from Network Volume to local NVMe..."
mkdir -p /root/data
# the below MUST match the filename of the compressed archive in RunPod
tar -xf /workspace/dataset_640.tar -C /root/data

echo "cloning YOLOMG-STW repository..."
git clone https://${GITHUB_PAT}@github.com/samt-w/yolomg-stw.git /root/YOLOMG
cd /root/YOLOMG

echo "installing dependencies..."
pip install -r requirements.txt

echo "starting YOLOMG Training - see ClearML for logs..."
python train.py --img 640 --batch 64 --epochs 100 --data /root/data/data.yaml --weights "" --cfg models/yolov5s.yaml --device 0
"""
### ----------------------------------------------------------------

### ------- HOST DEVICE TRIGGER SCRIPT -------------------------------------------------
print("Deploying training pod to RunPod...")

# base64 encoding to prevent quote escaping issues
encoded_script = base64.b64encode(pod_bash_script.encode()).decode()

try:
    pod = runpod.create_pod(
        name = "YOLOMG-Cloud-Trainer",
        image_name = "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04",
        gpu_type_id = GPU_ID,
        cloud_type = "SECURE",
        gpu_count = 1,
        # because using network volume, can set volume to 0
        volume_in_gb = 0,
        # set NVMe storage size
        container_disk_in_gb = CONTAINER_DISK_GB,
        network_volume_id = NETWORK_VOLUME_ID,
        env = pod_env_vars,
        docker_args = f"bash -c \"echo {encoded_script} | base64 -d | bash\""
    )

    print(f"Successfully booted Pod ID: {pod.get('id')}.")
    print("Training progress available on ClearML.")

except Exception as e:
    print("Failed to deploy pod.")
    print(f"Error details: {e}")