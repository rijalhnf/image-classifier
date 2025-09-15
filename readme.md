# Image Classifier & HS Code AI Application

## Overview
This application uses deep learning to classify images into predefined categories and can also suggest HS codes for products based on images. It leverages PyTorch for model inference and exposes the functionality through a FastAPI web service that allows users to upload and classify images, as well as predict HS codes.

## Datasets Used
- **Image Classification:** EfficientNet-B0 pre-trained on ImageNet.
- **HS Code Mapping:** [ronnieaban/hs-code](https://huggingface.co/datasets/ronnieaban/hs-code) dataset for HS code descriptions and matching.

## Features
- Upload images through a REST API endpoint
- Real-time image classification using EfficientNet-B0 (pre-trained on ImageNet)
- HS code suggestion endpoint for product images (demo mapping)
- Optimized for Apple Silicon (M1/M2) using MPS acceleration
- Simple API integration for frontend applications

## Technical Stack
- Python 3.10
- PyTorch with Apple MPS support
- FastAPI for API endpoints
- Docker for deployment (recommended)
- Conda environment for local development

## Getting Started

### Prerequisites
- macOS with Apple Silicon (M1/M2 chip) or Linux ARM device
- Miniforge/Conda installed (for local dev)
- VS Code (optional)

### Installation (Local Development)
1. Clone this repository
2. Activate the conda environment:
    ```bash
    conda activate ai
    ```
3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application (Local)
1. Start the FastAPI server:
    ```bash
    uvicorn app:app --reload
    ```
2. Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)
3. Use the `/classify` endpoint to upload and classify images
4. Use the `/predict-hs` endpoint to upload an image and get a suggested HS code (demo mapping)

## API Endpoints
- `POST /classify`: Upload an image for classification (returns top-5 categories)
- `POST /predict-hs`: Upload an image to get a predicted HS code (6-digit, demo mapping)
- `GET /`: API health check

## Model Information
This application uses a pre-trained EfficientNet-B0 model (ImageNet), optimized for running on Apple Silicon or ARM CPUs using PyTorch's MPS or CPU device.

## HS Code Prediction
The `/predict-hs` endpoint uses a demo mapping from ImageNet classes to HS codes. For real customs applications, a custom-trained model and dataset would be required.

## Quick command in terminal docker
```bash
sudo docker stop image-classifier 2>/dev/null || true
sudo docker rm -f image-classifier 2>/dev/null || true
git pull
sudo docker build -t image-classifier .
sudo docker run -d --restart unless-stopped \
  -p 8000:8000 \
  -e TORCH_NUM_THREADS=2 -e OMP_NUM_THREADS=2 \
  --name image-classifier image-classifier
```

## Additional Prototypes
Other AI prototypes (e.g., document validation, HS code recommendation, customs valuation anomaly detection) are available in the `prototypes/` folder for demo and experimentation.

---