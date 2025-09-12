import io
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

# Initialize FastAPI
app = FastAPI(
    title="Image Classifier API",
    description="An API for classifying images using a pre-trained ResNet model"
)

# Allowed FRONTEND origins only (do not include the API origin here)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://rij.al",
    "https://www.rij.al",
]

# Also validate upload content types
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}

# CORS middleware (handles OPTIONS preflight automatically)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model
def load_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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

    return model, device

# Load ImageNet class labels
def load_imagenet_labels():
    weights = ResNet18_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]
    return categories

# Globals
model, device = load_model()
categories = load_imagenet_labels()

# Preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Image Classification API is running"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=415,
            content={"error": "Unsupported media type. Allowed: JPEG, PNG."},
        )

    try:
        # Read bytes
        image_data = await file.read()
        if not image_data:
            return JSONResponse(status_code=400, content={"error": "Empty file."})

        # Open with PIL; ensure 3-channel RGB (handles PNG with alpha/palette)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)

        # Top-5 predictions
        top_probs, top_idxs = torch.topk(probs, 5, dim=1)
        top_probs = top_probs.squeeze(0).cpu().tolist()
        top_idxs = top_idxs.squeeze(0).cpu().tolist()

        results = [
            {"category": categories[idx], "probability": float(p)}
            for idx, p in zip(top_idxs, top_probs)
        ]

        return JSONResponse(
            status_code=200,
            content={"filename": file.filename, "predictions": results}
        )

    except UnidentifiedImageError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file. Provide a valid JPEG or PNG."},
        )
    except Exception:
        # Avoid leaking internals
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error."},
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)