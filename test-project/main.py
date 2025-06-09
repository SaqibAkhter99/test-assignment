# main.py

import base64
import io
from PIL import Image
import re

# --- CHANGE 1: Import the single, self-contained Model class ---
from model import Model

# This will hold the single, initialized model object, making it accessible
# to the run() function.
model = None

def setup():
    """
    This function is run once when the serverless machine boots up.
    It loads the self-contained model to be reused across all requests.
    """
    # Use the global keyword to modify the 'model' variable defined outside this function
    global model
    
    # --- CHANGE 2: Initialize only the single Model object ---
    model = Model()
    print("Setup complete: Self-contained Model is initialized and ready.")

def run(item: dict):
    """
    This function is run for every prediction request.
    It decodes the image and passes it to the model for a full end-to-end prediction.
    """
    if 'image_b64' not in item:
        return {"error": "Request must include 'image_b64' field."}

    try:
        # This part correctly decodes the incoming data
        base64_string = item['image_b64']
        base64_string = re.sub('^data:image/.+;base64,', '', base64_string)
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))

        # --- CHANGE 3: Simplify the prediction logic ---
        # The run() function's job is now just to prepare the payload
        # and call the model's predict method.
        
        # 1. Create the payload our model's predict() method expects
        payload = {"image": image}
        
        # 2. Call predict() and get the final integer ID directly
        predicted_class_id = model.predict(payload)

        # 3. Return the simple, JSON-friendly result
        return {
            "predicted_class_id": predicted_class_id
        }

    except Exception as e:
        print(f"!!! INFERENCE ERROR: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}
