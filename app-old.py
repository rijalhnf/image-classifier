import torch
import torchvision.models as models
import argparse
from utils import load_imagenet_labels, preprocess_image, get_device

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Classification with ResNet50")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to local image file")
    group.add_argument("--url", help="URL to image file")
    args = parser.parse_args()
    
    # Get the device (MPS for M1 Mac)
    device = get_device()
    print(f"Using device: {device}")
    
    # Load pre-trained ResNet model
    print("Loading pre-trained ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.to(device)
    model.eval()
    
    # Load class labels
    classes = load_imagenet_labels()
    
    # Preprocess the image
    if args.image:
        print(f"Processing local image: {args.image}")
        input_tensor = preprocess_image(image_path=args.image)
    else:
        print(f"Processing image from URL: {args.url}")
        input_tensor = preprocess_image(image_url=args.url)
    
    # Move input to the same device as model
    input_tensor = input_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Process results
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    # Print results
    print("\nTop 5 Predictions:")
    print("-----------------")
    for i in range(5):
        print(f"{classes[top5_indices[i]]}: {top5_prob[i].item()*100:.2f}%")

if __name__ == "__main__":
    main()