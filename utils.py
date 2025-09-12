import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import json

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