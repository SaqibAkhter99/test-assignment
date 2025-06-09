from PIL import Image
import numpy as np

# --- CHANGE 1: Import the single, self-contained Model class ---
from model import Model

# --- Configuration ---
# Choose which image you want to test. Note: The expected class for tiger shark is 2.
IMAGE_TO_TEST = "n01491361_tiger_shark.JPEG"
EXPECTED_ID = 2 
# IMAGE_TO_TEST = "n01667114_mud_turtle.jpeg"
# EXPECTED_ID = 35

print(f"--- Starting Local Test for '{IMAGE_TO_TEST}' ---")

try:
    # --- CHANGE 2: Initialize only the single Model object ---
    # This single object contains both the preprocessor and the ONNX session.
    print("Step 1: Initializing the self-contained Model...")
    model = Model()
    print("Initialization complete.")

    # 2. Load the image from the file (this step remains the same)
    print(f"\nStep 2: Loading image '{IMAGE_TO_TEST}'...")
    img = Image.open(IMAGE_TO_TEST)
    print("Image loaded successfully.")

    # --- CHANGE 3: Call the model's predict method directly ---
    # We no longer do the preprocessing here. The model handles it internally.
    # The predict method now expects a dictionary with the raw PIL image.
    print("\nStep 3: Running end-to-end prediction...")
    # This structure exactly matches what the Cerebrium server will do.
    prediction_payload = {"image": img}
    predicted_id = model.predict(prediction_payload)
    print("Prediction complete.")

    # 4. Print and verify the final result
    print("\n--- TEST RESULT ---")
    print(f"Predicted Class ID = {predicted_id}")
    
    assert predicted_id == EXPECTED_ID, f"FAILED. Predicted {predicted_id}, expected {EXPECTED_ID}."
    print("✅ PASSED: Prediction is correct.")


except Exception as e:
    print("\n--- AN ERROR OCCURRED ---")
    print(f"Error: {e}")

