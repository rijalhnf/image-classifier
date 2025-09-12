# Image Classifier AI Application

## Overview
This application uses deep learning to classify images into predefined categories. It leverages PyTorch for model training and inference, and exposes the functionality through a FastAPI web service that allows users to upload and classify images.

## Features
- Upload images through a REST API endpoint
- Real-time image classification using a pre-trained CNN model
- Optimized for Apple Silicon (M1/M2) using MPS acceleration
- Simple API integration for frontend applications

## Technical Stack
- Python 3.10
- PyTorch with Apple MPS support
- FastAPI for API endpoints
- Conda environment for dependency management

## Getting Started

### Prerequisites
- macOS with Apple Silicon (M1/M2 chip)
- Miniforge/Conda installed
- VS Code (optional)

### Installation
1. Clone this repository
2. Activate the conda environment:
```bash
conda activate ai
```
3. Install required packages:
```bash
pip install fastapi uvicorn python-multipart
```

### Running the Application
1. Start the FastAPI server:
```bash
uvicorn app:app --reload
```
2. Access the API documentation at http://localhost:8000/docs
3. Use the /classify endpoint to upload and classify images

## API Endpoints
- `POST /classify`: Upload an image for classification
- `GET /`: API health check

## Model Information
This application uses a pre-trained ResNet-18 model fine-tuned on ImageNet dataset, optimized for running on Apple Silicon using PyTorch's MPS device.

## Quick command in terminal docker
git pull
sudo docker build -t image-classifier .
sudo docker stop $(sudo docker ps -q --filter "ancestor=image-classifier") 2>/dev/null || true
sudo docker run -d -p 8000:8000 --name image-classifier image-classifier