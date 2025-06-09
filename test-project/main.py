# main.py

import base64
import io
from PIL import Image
import re
from model import Model

model = None

def setup():
    """
    Loads the self-contained model. If this fails, the server log will
    show the specific error from the Model.__init__ method.
    """
    global model
    try:
        model = Model()
        print("Setup complete: Self-contained Model is initialized and ready.")
    except Exception as e:
        print(f"!!! MODEL INITIALIZATION FAILED: {e}")
        # 'model' will remain None

def run(item: dict):
    """
    Runs prediction. Now includes a check to ensure the model loaded correctly.
    """
    # --- THIS IS THE KEY CHANGE ---
    # Check if the model failed to initialize during setup()
    if model is None:
        return {"error": "Model object is None. Check setup logs for initialization errors."}

    if 'image_b64' not in item:
        return {"error": "Request must include 'image_b64' field."}

    try:
        base64_string = item['image_b64']
        base64_string = re.sub('^data:image/.+;base64,', '', base64_string)
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        payload = {"image": image}
        predicted_class_id = model.predict(payload)

        return {"predicted_class_id": predicted_class_id}

    except Exception as e:
        print(f"!!! INFERENCE ERROR: {e}")
        return {"error": f"An unexpected error occurred during run: {str(e)}"}
