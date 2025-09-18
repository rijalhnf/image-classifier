
# --- Imports ---
# torch: Deep learning library for model inference
# torchvision: Image transforms for preprocessing
# PIL: Image loading and manipulation
# requests: Download files/images from the web
# BytesIO: Handle image bytes in memory
# json, csv: Data file handling
# rapidfuzz: Fast fuzzy string matching for HS code search
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import json
import csv
from rapidfuzz import process, fuzz


# --- Global Variables ---
HS_DATASET_PATH = "hs_code_dataset.csv"  # Path to HS code dataset (CSV file)


# --- ImageNet Class Labels ---
def load_imagenet_labels():
    """
    Downloads and returns the list of ImageNet class labels.
    Used to map model predictions to human-readable categories.
    """
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(labels_url)
    categories = response.text.splitlines()
    return categories


# --- Image Preprocessing ---
def preprocess_image(image_path=None, image_url=None):
    """
    Loads and preprocesses an image for model input.
    Accepts either a local file path or a URL.
    Returns a PyTorch tensor ready for EfficientNet/ImageNet models.
    """
    if image_path:
        img = Image.open(image_path)
    elif image_url:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    else:
        raise ValueError("Either image_path or image_url must be provided")
    # Standard transforms for pre-trained models
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).unsqueeze(0)  # Add batch dimension


# --- Device Selection ---
def get_device():
    """
    Returns the best available device for PyTorch (Apple M1, CUDA GPU, or CPU).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# --- ImageNet to HS Code Mapping (Demo) ---
IMAGENET_TO_HS = {
    # Maps common ImageNet labels to example HS codes and descriptions
    "smartphone": ("851713", "Smartphones"),
    "cellular telephone": ("851714", "Other telephones for cellular networks"),
    "cordless phone": ("851711", "Line telephone sets with cordless handsets"),
    "modem": ("851762", "Machines for transmission or regeneration of data"),
    "router": ("851743", "Control and adaptor units, including routers"),
    "laptop": ("847130", "Portable digital automatic data processing machines"),
    # Add more mappings as needed
}


def get_hs_code(imagenet_label: str):
    """
    Maps an ImageNet label to a demo HS code and description.
    Returns (HS code, description) tuple, or ('000000', 'Unknown...') if not found.
    """
    for key in IMAGENET_TO_HS:
        if key in imagenet_label.lower():
            return IMAGENET_TO_HS[key]
    return ("000000", "Unknown or unmapped product")


# --- HS Code Dataset Loader ---
def load_hs_dataset():
    """
    Loads HS code dataset from CSV file into memory.
    Each row should have a 'text' column: 'HS_CODE - Description'.
    Returns a list of dicts: {hs_code, hs_description}
    """
    hs_rows = []
    with open(HS_DATASET_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Split text to get hs_code and description
            text = row["text"]
            if " - " in text:
                hs_code, description = text.split(" - ", 1)
            else:
                hs_code, description = text, ""
            hs_rows.append({
                "hs_code": hs_code.strip(),
                "hs_description": description.strip(),
            })
    return hs_rows

# Load dataset once at import for fast access
HS_DATASET = load_hs_dataset()


# --- Fuzzy Search for HS Codes ---
def search_hs_code(label, limit=3):
    """
    Finds the closest HS code descriptions to a given label using fuzzy matching.
    Returns a list of top matches with HS code, description, and match score.
    """
    results = process.extract(
        label,
        [row["hs_description"] for row in HS_DATASET],
        scorer=fuzz.WRatio,
        limit=limit
    )
    hs_matches = []
    for match in results:
        desc, score, idx = match
        hs_row = HS_DATASET[idx]
        hs_matches.append({
            "hs_code": hs_row["hs_code"],
            "hs_description": hs_row["hs_description"],
            "match_score": score
        })
    return hs_matches