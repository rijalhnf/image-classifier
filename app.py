
# === IMAGE CLASSIFIER & HS CODE SUGGESTION API ===
# This is a Python FastAPI web application that uses AI to classify images and suggest HS codes for customs.

# --- Imports ---
# Standard library imports
import io  # For handling byte streams (image data)
import os  # For environment variables and system settings

# AI/ML imports
import torch  # PyTorch: deep learning library
import torchvision.transforms as transforms  # Image preprocessing
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights  # Pre-trained model
import torch.nn.functional as F  # For softmax and other functions

# Image handling
from PIL import Image, UnidentifiedImageError  # For opening and validating images

# FastAPI imports
from fastapi import FastAPI, UploadFile, File  # Web API framework and file upload
from fastapi.middleware.cors import CORSMiddleware  # For cross-origin requests (frontend integration)
from fastapi.responses import JSONResponse  # For sending JSON responses

# Utility functions for HS code matching
from utils import get_hs_code, search_hs_code

# --- FastAPI App Setup ---
app = FastAPI(
    title="Image Classifier API",
    description="An API for classifying images using a pre-trained EfficientNet-B0 model and suggesting HS codes."
)

# --- CORS Settings ---
# Allow requests from these frontend origins (for browser apps)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://rij.al",
    "https://www.rij.al",
]

# --- File Type and Size Settings ---
# Accept common image types (JPG, PNG)
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/x-png"
}
# Limit upload size to 10 MB (can be changed)
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))

# --- Optional: Limit CPU threads for stability on small devices ---
try:
    tn = int(os.getenv("TORCH_NUM_THREADS", "2"))
    torch.set_num_threads(max(1, tn))
except Exception:
    pass

# --- Enable CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading Function ---
def load_model():
    """
    Loads the EfficientNet-B0 model pre-trained on ImageNet.
    Selects the best device (Apple MPS, CUDA GPU, or CPU).
    Returns the model, device, and weights object.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1  # Pre-trained weights
    model = efficientnet_b0(weights=weights)  # Load model
    model.eval()  # Set to evaluation mode (no training)
    # Select device: Apple MPS (Mac), CUDA (GPU), or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return model, device, weights

# --- Load ImageNet Class Labels ---
def load_imagenet_labels(w):
    """
    Returns the list of category names for ImageNet classes.
    """
    return w.meta["categories"]

# --- Initialize Model, Device, Labels, and Preprocessing ---
model, device, weights = load_model()
categories = load_imagenet_labels(weights)
preprocess = weights.transforms()  # Recommended preprocessing for EfficientNet-B0

# --- Health Check Endpoint ---
@app.get("/")
def health_check():
    """
    Simple endpoint to check if the API is running.
    """
    return {"status": "ok", "message": "Image Classification API is running"}

# --- Image Classification Endpoint ---
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Receives an image file, classifies it using EfficientNet-B0, and returns top-5 predicted categories with probabilities.
    """
    # --- Validate file type ---
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=415,
            content={"error": "Unsupported media type. Allowed: JPEG, PNG."},
        )

    try:
        data = await file.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "Empty file."})

        if len(data) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)} MB."},
            )

        # --- Open and validate image ---
        try:
            img = Image.open(io.BytesIO(data))
        except UnidentifiedImageError:
            return JSONResponse(status_code=400, content={"error": "Invalid image file."})

        # --- Check image format ---
        if img.format not in ("JPEG", "PNG"):
            return JSONResponse(
                status_code=415,
                content={"error": "Unsupported image format. Allowed: JPEG, PNG."},
            )

        # --- Convert to RGB if needed ---
        if img.mode != "RGB":
            img = img.convert("RGB")

        # --- Preprocess and run model ---
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)

        # --- Get top-5 predictions ---
        top_probs, top_idxs = torch.topk(probs, 5, dim=1)
        top_probs = top_probs.squeeze(0).cpu().tolist()
        top_idxs = top_idxs.squeeze(0).cpu().tolist()

        results = [
            {"category": categories[idx], "probability": float(p)}
            for idx, p in zip(top_idxs, top_probs)
        ]

        # --- Return results as JSON ---
        return JSONResponse(status_code=200, content={
            "filename": file.filename,
            "predictions": results
        })

    except Exception:
        return JSONResponse(status_code=500, content={"error": "Internal server error."})

# --- HS Code Suggestion Endpoint ---
@app.post("/predict-hs")
async def predict_hs(file: UploadFile = File(...)):
    """
    Receives an image file, classifies it, and suggests relevant HS codes by fuzzy matching predicted labels to HS code descriptions.
    Returns top-3 predicted labels, each with top-3 HS code matches.
    """
    # --- Validate file type ---
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=415,
            content={"error": "Unsupported media type. Allowed: JPEG, PNG."},
        )
    try:
        data = await file.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "Empty file."})

        if len(data) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)} MB."},
            )

        # --- Open and validate image ---
        try:
            img = Image.open(io.BytesIO(data))
        except UnidentifiedImageError:
            return JSONResponse(status_code=400, content={"error": "Invalid image file."})

        # --- Convert to RGB if needed ---
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)

        # --- Get top-3 predictions ---
        top_probs, top_idxs = torch.topk(probs, 3, dim=1)
        top_probs = top_probs.squeeze(0).cpu().tolist()
        top_idxs = top_idxs.squeeze(0).cpu().tolist()

        results = []
        for idx, prob in zip(top_idxs, top_probs):
            label = categories[idx]  # Predicted product name
            # Fuzzy match label to HS code descriptions
            hs_matches = search_hs_code(label, limit=3)
            hs_list = [
                {
                    "hs_code": hs["hs_code"],
                    "hs_description": hs["hs_description"],
                    "match_score": hs["match_score"]
                }
                for hs in hs_matches
            ]
            results.append({
                "predicted_label": label,
                "hs": hs_list,
                "probability": float(prob)
            })

        # --- Return grouped results ---
        return JSONResponse(status_code=200, content={
            "filename": file.filename,
            "predictions": results
        })

    except Exception:
        return JSONResponse(status_code=500, content={"error": "Internal server error."})

# --- Main block for running locally (not needed in Docker) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

app = FastAPI(
    title="Image Classifier API",
    description="An API for classifying images using a pre-trained EfficientNet-B0 model"
)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://rij.al",
    "https://www.rij.al",
]

# Accept common headers the browser may send for PNG/JPEG
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg",
    "image/png", "image/x-png",  # some browsers use image/x-png
}

# Optional: cap upload size (bytes)
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10 MB

# Optional: limit CPU threads for stability on small devices
try:
    tn = int(os.getenv("TORCH_NUM_THREADS", "2"))
    torch.set_num_threads(max(1, tn))
except Exception:
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.eval()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return model, device, weights

def load_imagenet_labels(w):
    return w.meta["categories"]

model, device, weights = load_model()
categories = load_imagenet_labels(weights)

# Use the model’s recommended preprocessing
preprocess = weights.transforms()

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Image Classification API is running"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Fast reject on clearly wrong content-types (still validate by PIL below)
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=415,
            content={"error": "Unsupported media type. Allowed: JPEG, PNG."},
        )

    try:
        data = await file.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "Empty file."})

        if len(data) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)} MB."},
            )

        # Open image and ensure RGB
        try:
            img = Image.open(io.BytesIO(data))
        except UnidentifiedImageError:
            return JSONResponse(status_code=400, content={"error": "Invalid image file."})

        # Validate actual format
        if img.format not in ("JPEG", "PNG"):
            return JSONResponse(
                status_code=415,
                content={"error": "Unsupported image format. Allowed: JPEG, PNG."},
            )

        if img.mode != "RGB":
            img = img.convert("RGB")

        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)

        top_probs, top_idxs = torch.topk(probs, 5, dim=1)
        top_probs = top_probs.squeeze(0).cpu().tolist()
        top_idxs = top_idxs.squeeze(0).cpu().tolist()

        results = [
            {"category": categories[idx], "probability": float(p)}
            for idx, p in zip(top_idxs, top_probs)
        ]

        return JSONResponse(status_code=200, content={
            "filename": file.filename,
            "predictions": results
        })

    except Exception:
        return JSONResponse(status_code=500, content={"error": "Internal server error."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

@app.post("/predict-hs")
async def predict_hs(file: UploadFile = File(...)):
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=415,
            content={"error": "Unsupported media type. Allowed: JPEG, PNG."},
        )
    try:
        data = await file.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "Empty file."})

        if len(data) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)} MB."},
            )

        try:
            img = Image.open(io.BytesIO(data))
        except UnidentifiedImageError:
            return JSONResponse(status_code=400, content={"error": "Invalid image file."})

        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)

        top_probs, top_idxs = torch.topk(probs, 3, dim=1)
        top_probs = top_probs.squeeze(0).cpu().tolist()
        top_idxs = top_idxs.squeeze(0).cpu().tolist()

        results = []
        for idx, prob in zip(top_idxs, top_probs):
            label = categories[idx]
            hs_matches = search_hs_code(label, limit=3)
            hs_list = [
                {
                    "hs_code": hs["hs_code"],
                    "hs_description": hs["hs_description"],
                    "match_score": hs["match_score"]
                }
                for hs in hs_matches
            ]
            results.append({
                "predicted_label": label,
                "hs": hs_list,
                "probability": float(prob)
            })

        return JSONResponse(status_code=200, content={
            "filename": file.filename,
            "predictions": results
        })

    except Exception:
        return JSONResponse(status_code=500, content={"error": "Internal server error."})