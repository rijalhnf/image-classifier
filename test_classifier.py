import requests
import json
import sys

def test_classifier(image_path):
    """Test the image classifier API by sending an image to the endpoint."""
    print(f"Testing with image: {image_path}")
    
    # API endpoint
    url = "http://localhost:8000/classify"
    
    try:
        # Open the image file
        with open(image_path, 'rb') as f:
            # Create the files parameter for the request
            files = {'file': (image_path, f, 'image/jpeg')}
            
            # Make the request to the API
            response = requests.post(url, files=files)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                result = response.json()
                
                # Print the predictions
                print("\nClassification Results:")
                print(f"Filename: {result['filename']}")
                print("\nTop 5 predictions:")
                
                for i, prediction in enumerate(result['predictions'], 1):
                    print(f"{i}. {prediction['category']}: {prediction['probability']:.4f}")
                    
                return True
            else:
                print(f"Error: API returned status code {response.status_code}")
                print(response.text)
                return False
    
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Check if an image path was provided
    if len(sys.argv) < 2:
        print("Usage: python test_classifier.py <path_to_image>")
        sys.exit(1)
    
    # Get the image path from command line arguments
    image_path = sys.argv[1]
    
    # Test the classifier
    test_classifier(image_path)