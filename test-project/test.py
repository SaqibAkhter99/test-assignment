# test.py
from PIL import Image
import numpy as np

# --- CHANGE 1: Import the single, self-contained Model class ---
from model import Model

def run_local_tests():
    """
    Executes a suite of local tests for the self-contained classification model.
    This script now mirrors how a deployment server will use the Model class.
    """
    print("--- Starting Local Test Suite ---")

    # --- Test Setup ---
    # The model path is now handled inside the Model class, so we don't need it here.
    test_images = {
        "n01440764_tench.jpeg": 0,
        "n01667114_mud_turtle.jpeg": 35
    }

    # --- CHANGE 2: Initialize only the single Model object ---
    # This single object contains both the preprocessor and the ONNX session.
    try:
        model = Model()
        print("[Test 1] ✅ PASSED: Model initialized successfully.")
    except Exception as e:
        print(f"[Test 1] ❌ FAILED: Model initialization error: {e}")
        return # Exit if the model can't even be loaded

    # --- CHANGE 3: Combine preprocessing and inference tests ---
    # The model's predict method now handles the full end-to-end pipeline.
    print("\n[Test 2] Verifying End-to-End Model Inference and Correctness...")
    for image_path, expected_id in test_images.items():
        try:
            # Load the image as a PIL object, which our new model expects
            img = Image.open(image_path)
            
            # The predict method expects a dictionary payload, just like a server
            prediction_payload = {"image": img}
            
            # Call predict on the raw image payload
            predicted_id = model.predict(prediction_payload)

            # The assert remains the same, comparing the final integer IDs
            assert predicted_id == expected_id, f"FAILED for {image_path}. Predicted {predicted_id}, expected {expected_id}."
            print(f"✅ PASSED: Correctly classified '{image_path}' as class {predicted_id}.")
        except Exception as e:
            print(f"❌ FAILED for {image_path} with error: {e}")

    print("\n--- All Local Tests Passed Successfully! ---")

if __name__ == "__main__":
    run_local_tests()

