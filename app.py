import io
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Initialize FastAPI
app = FastAPI(title="Image Classifier API", 
              description="An API for classifying images using a pre-trained ResNet model")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React app's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load pre-trained model
def load_model():
    # Use pre-trained ResNet-18 model
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    
    # Move to MPS device if available (Apple Silicon optimization)
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
    # Download the ImageNet class labels
    weights = ResNet18_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]
    return categories

# Global variables
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
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Preprocess the image
    image_tensor = preprocess(image)
    
    # Add batch dimension and move to appropriate device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        
    # Get top 5 predictions
    _, indices = torch.topk(outputs, 5)
    indices = indices.cpu().numpy()[0]
    
    # Return class predictions
    results = [
        {"category": categories[idx], "probability": float(outputs[0, idx])}
        for idx in indices
    ]
    
    return JSONResponse(content={
        "filename": file.filename,
        "predictions": results
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)