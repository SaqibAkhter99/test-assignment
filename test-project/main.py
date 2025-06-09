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
    try:
        # ... (your existing try block logic) ...
        image_data = base64.b64decode(item['image_b64'])
        image = Image.open(io.BytesIO(image_data))
        preprocessed_img = preprocessor.preprocess(image)
        probabilities = model.predict(preprocessed_img)
        
        # Add a check here to ensure probabilities are not empty
        if probabilities is None or probabilities.size == 0:
            raise ValueError("Model returned an empty prediction.")

        predicted_class_id = int(np.argmax(probabilities))
        return {
            "predicted_class_id": predicted_class_id,
            "probabilities": probabilities.tolist()
        }

    except Exception as e:
        # CRUCIAL CHANGE: Print the actual error to the server logs
        print(f"!!! INFERENCE ERROR: {e}")
        # Return a dictionary that clearly indicates an error
        return {"error": f"An unexpected error occurred: {str(e)}"}
