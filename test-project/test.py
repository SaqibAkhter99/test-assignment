# test.py
import numpy as np
import torch
from pytorch_model import ResNet # Used for numerical comparison
from model import ImagePreprocessor, OnnxModel

def run_local_tests():
    """
    Executes a comprehensive suite of local tests for the image classification pipeline.
    """
    print("--- Starting Local Test Suite ---")

    # --- Test Setup ---
    weights_path = 'weights/pytorch_model_weights.pth'
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
        probabilities = onnx_model.predict(processed_img)
        predicted_id = np.argmax(probabilities)
        assert predicted_id == expected_id, f"FAILED for {image_path}. Predicted {predicted_id}, expected {expected_id}."
        print(f"✅ PASSED: Correctly classified '{image_path}' as class {predicted_id}.")
    
    # --- Test 3: Numerical Consistency Check (ONNX vs. PyTorch) ---
    print("\n[Test 3] Verifying Numerical Consistency between ONNX and PyTorch...")
    # Load original PyTorch model
    pt_model = ResNet(num_classes=1000)
    pt_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    pt_model.eval()

    # Get outputs from both models using the same input
    sample_input_tensor = torch.from_numpy(preprocessor.preprocess(sample_image_path))
    with torch.no_grad():
        pt_output = pt_model(sample_input_tensor).numpy()
    
    onnx_output = onnx_model.predict(sample_input_tensor.numpy())
    
    # Check if outputs are numerically close
    np.testing.assert_allclose(pt_output, onnx_output, rtol=1e-03, atol=1e-05)
    print("✅ PASSED: ONNX and PyTorch model outputs are numerically consistent.")

    print("\n--- All Local Tests Passed Successfully! ---")

if __name__ == "__main__":
    run_local_tests()

