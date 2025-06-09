# convert_to_onnx.py
import torch
import os
from pytorch_model import Classifier
 # Assumes pytorch_model.py is in the same directory

def convert_model_to_onnx(weights_path, output_path):
    """
    Loads a PyTorch model with its weights and converts it to the ONNX format.

    Args:
        weights_path (str): The path to the .pth file containing the model's weights.
        output_path (str): The path where the converted .onnx model will be saved.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found. Please download it and place it at: {weights_path}")

    # Initialize the model architecture (ImageNet has 1000 classes)
    model = Classifier(num_classes=1000)

    # Load the trained weights into the model structure
    # Use map_location='cpu' to ensure it works regardless of the machine's hardware
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    # Set the model to evaluation mode. This is crucial for correct inference behavior [13].
    model.eval()
    print("PyTorch model loaded successfully and set to evaluation mode.")

    # Create a dummy input tensor that matches the model's expected input shape
    # The shape is [batch_size, channels, height, width]
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    # Export the model to ONNX format [2]
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,        # Store the trained parameter weights inside the model file
        opset_version=11,          # The ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=['input'],     # The model's input names
        output_names=['output'],   # The model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # Allow for variable batch size
                      'output' : {0 : 'batch_size'}}
    )
    print(f"Model successfully converted to ONNX and saved at: {output_path}")

if __name__ == "__main__":
    pytorch_weights_file = 'weights/pytorch_model_weights.pth'
    onnx_model_file = 'model.onnx'
    
    convert_model_to_onnx(pytorch_weights_file, onnx_model_file)