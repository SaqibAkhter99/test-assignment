# test.py
from model import ImagePreprocessor, OnnxModel

def run_local_tests():
    """
    Executes a comprehensive suite of local tests for the image classification pipeline.
    """
    print("--- Starting Local Test Suite ---")

    # --- Test Setup ---
    onnx_path = 'model.onnx'
    test_images = {
        "n01440764_tench.jpeg": 0,
        "n01667114_mud_turtle.jpeg": 35
    }
    preprocessor = ImagePreprocessor()
    onnx_model = OnnxModel(onnx_path)

    # --- Test 1: Image Pre-processing Logic ---
    print("\n[Test 1] Verifying Image Pre-processing...")
    sample_image_path = list(test_images.keys())[0]
    processed_img = preprocessor.preprocess(sample_image_path)
    expected_shape = (1, 3, 224, 224)
    assert processed_img.shape == expected_shape, f"Pre-processing failed. Shape is {processed_img.shape}, expected {expected_shape}."
    print("✅ PASSED: Pre-processing output shape is correct.")

    # --- Test 2: ONNX Model Inference and Correctness ---
    print("\n[Test 2] Verifying ONNX Model Inference and Correctness...")
    for image_path, expected_id in test_images.items():
        processed_img = preprocessor.preprocess(image_path)
        
        # --- THIS IS THE FIX ---
        # The predict method now directly returns the final ID.
        predicted_id = onnx_model.predict(processed_img)

        # The assert remains the same, but now it compares two integers.
        assert predicted_id == expected_id, f"FAILED for {image_path}. Predicted {predicted_id}, expected {expected_id}."
        print(f"✅ PASSED: Correctly classified '{image_path}' as class {predicted_id}.")
    print("\n--- All Local Tests Passed Successfully! ---")

if __name__ == "__main__":
    run_local_tests()

