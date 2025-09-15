import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import json
import csv
from rapidfuzz import process, fuzz

HS_DATASET_PATH = "hs_code_dataset.csv"  # Make sure this file exists in your project root

# Load ImageNet class labels
def load_imagenet_labels():
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(labels_url)
    categories = response.text.splitlines()
    return categories

# Prepare image for model input
def preprocess_image(image_path=None, image_url=None):
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

# Get device optimized for M1
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
    # Simple mapping for demo purposes
IMAGENET_TO_HS = {
    "smartphone": ("851713", "Smartphones"),
    "cellular telephone": ("851714", "Other telephones for cellular networks"),
    "cordless phone": ("851711", "Line telephone sets with cordless handsets"),
    "modem": ("851762", "Machines for transmission or regeneration of data"),
    "router": ("851743", "Control and adaptor units, including routers"),
    "laptop": ("847130", "Portable digital automatic data processing machines"),
    # Add more mappings as needed
}

def get_hs_code(imagenet_label: str):
    # Lowercase and match
    for key in IMAGENET_TO_HS:
        if key in imagenet_label.lower():
            return IMAGENET_TO_HS[key]
    return ("000000", "Unknown or unmapped product")

# Load HS code dataset into memory
def load_hs_dataset():
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

HS_DATASET = load_hs_dataset()

def search_hs_code(label, limit=3):
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