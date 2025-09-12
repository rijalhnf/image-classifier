import io
import os
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn.functional as F

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