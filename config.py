from pathlib import Path

# ================= MODEL =================
MODEL_NAME = "Hybrid Swin-UNet Brain Tumor Classifier"
MODEL_PATH = "best_model.pth"

# ================= CLASS MAPPING =================
MODEL_CLASS_NAMES = [
    "no_tumor",
    "glioma",
    "meningioma",
    "pituitary"
]

DISPLAY_CLASS_NAMES = [
    "No Tumor",
    "Glioma",
    "Meningioma",
    "Pituitary Tumor"
]

# ================= IMAGE SETTINGS =================
IMAGE_SIZE = 224
IMG_SIZE = 224

# ================= NORMALIZATION (FIXED - REQUIRED BY preprocess.py) =================
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ================= CONFIDENCE =================
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.40

# ================= HISTORY =================
HISTORY_FILE = Path("history.json")
HISTORY_IMAGES_DIR = Path("history_images")