import base64
import io
import numpy as np
from PIL import Image
import re
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
    if 'image_b64' not in item:
        return {"error": "Request must include 'image_b64' field."}

    try:
        base64_string = item['image_b64']
        
        # --- Step 2: Sanitize the base64 string ---
        # This regex removes "data:image/jpeg;base64," or similar prefixes
        base64_string = re.sub('^data:image/.+;base64,', '', base64_string)

        # Decode the sanitized base64 string
        image_data = base64.b64decode(base64_string)
        
        # Open the image from the in-memory bytes
        image = Image.open(io.BytesIO(image_data))

        preprocessed_img = preprocessor.preprocess(image)
        probabilities = model.predict(preprocessed_img)
        
        if probabilities is None or probabilities.size == 0:
            raise ValueError("Model returned an empty prediction.")

        predicted_class_id = int(np.argmax(probabilities))
        
        return {
            "predicted_class_id": predicted_class_id,
            "probabilities": probabilities.tolist()
        }

    except Exception as e:
        print(f"!!! INFERENCE ERROR: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}
