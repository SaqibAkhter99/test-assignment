
# main.py
import base64
import io
import numpy as np
from PIL import Image
from model import ImagePreprocessor, OnnxModel

def setup():
    """
    This function is run once when the serverless machine boots up.
    It loads the model and preprocessor to be reused across requests.
    """
    global preprocessor, model
    preprocessor = ImagePreprocessor()
    model = OnnxModel()
    print("Setup complete: Preprocessor and ONNX model are initialized and ready.")

def run(item: dict):
    """
    This function is called for every API request.
    It expects a JSON payload with a base64 encoded image string.
    """
    # Input validation
    if 'image_b64' not in item:
        return {"error": "Request must include 'image_b64' field."}

    try:
        # Decode the base64 string to bytes
        image_data = base64.b64decode(item['image_b64'])
        # Open the image from the in-memory bytes
        image = Image.open(io.BytesIO(image_data))

        # Process the image using our preprocessor class
        preprocessed_img = preprocessor.preprocess(image)

        # Get prediction probabilities from our model class
        probabilities = model.predict(preprocessed_img)
        
        # Get the index of the highest probability, which is the class ID
        predicted_class_id = int(np.argmax(probabilities))

        # Return the class ID and the full probability array
        return {
            "predicted_class_id": predicted_class_id,
            "probabilities": probabilities.tolist()
        }

    except Exception as e:
        # Return a server error if anything goes wrong
        return {"error": f"An unexpected error occurred: {str(e)}"}

