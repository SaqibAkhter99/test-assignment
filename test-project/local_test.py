# local_test.py

from PIL import Image
import numpy as np

# Import the classes from your model.py file
from model import ImagePreprocessor, OnnxModel

# --- Configuration ---
# Choose which image you want to test
IMAGE_TO_TEST = "n01667114_mud_turtle.jpeg" # Expected class: 0
# IMAGE_TO_TEST = "n01667114_mud_turtle.jpeg" # Expected class: 35

print(f"--- Starting Local Test for '{IMAGE_TO_TEST}' ---")

try:
    # 1. Initialize your preprocessor and model
    # This is exactly what the server would do on startup.
    print("Step 1: Initializing ImagePreprocessor and OnnxModel...")
    preprocessor = ImagePreprocessor()
    model = OnnxModel(model_path="model.onnx")
    print("Initialization complete.")

    # 2. Load the image from the file
    print(f"\nStep 2: Loading image '{IMAGE_TO_TEST}'...")
    img = Image.open(IMAGE_TO_TEST)
    print("Image loaded successfully.")

    # 3. Preprocess the image
    print("\nStep 3: Preprocessing image...")
    preprocessed_image = preprocessor.preprocess(img)
    
    # CRITICAL DEBUG STEP: Check the shape and type of the input tensor
    print(f"-> DEBUG: Preprocessed tensor shape: {preprocessed_image.shape}")
    print(f"-> DEBUG: Preprocessed tensor dtype: {preprocessed_image.dtype}")
    
    # Verify the shape is (1, 3, 224, 224) as required by ResNet models
    assert preprocessed_image.shape == (1, 3, 224, 224), "Shape is incorrect!"
    assert preprocessed_image.dtype == np.float32, "Data type is incorrect!"
    print("Preprocessing successful.")

    # 4. Get the prediction from the model
    print("\nStep 4: Running prediction...")
    prediction = model.predict(preprocessed_image)
    print("Prediction complete.")

    # 5. Print the final result
    print("\n--- TEST RESULT ---")
    print(f"Predicted Class ID = {prediction}")

except Exception as e:
    print("\n--- AN ERROR OCCURRED ---")
    print(f"Error: {e}")

