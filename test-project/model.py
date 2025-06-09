import numpy as np
from PIL import Image
import onnxruntime as ort

class ImagePreprocessor:
    """
    Handles all pre-processing steps required to transform an input image
    into a tensor suitable for the classification model.
    """
    def __init__(self, size=(224, 224)):
        self.size = size
        # Normalization values for ImageNet in [R, G, B] order
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def preprocess(self, image_input):
        """
        Takes a file path or PIL Image, performs all necessary pre-processing,
        and returns a numpy tensor.
        """
        if isinstance(image_input, str): # if a file path is provided
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image): # if a PIL image is provided
            img = image_input
        else:
            raise TypeError("Input must be a file path string or a PIL Image object.")

        # 1. Convert to RGB format
        img = img.convert("RGB")
        
        # 2. Resize to 224x224 using bilinear interpolation
        img = img.resize(self.size, Image.BILINEAR)

        # 3. Convert to numpy array and divide by 255 to scale to [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0

        # 4. Normalize by subtracting mean and dividing by standard deviation
        normalized_img = (img_np - self.mean) / self.std

        # 5. Transpose from [H, W, C] to [C, H, W] as expected by PyTorch models
        transposed_img = normalized_img.transpose((2, 0, 1))

        # 6. Add a batch dimension to create a [1, C, H, W] tensor
        return np.expand_dims(transposed_img, axis=0).astype(np.float32)

# In model.py, inside the OnnxModel class
class OnnxModel:
    def __init__(self, model_path="model.onnx"):
        print("Initializing ONNX Runtime session...")
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Get input details
            self.input_details = self.session.get_inputs()[0]
            self.input_name = self.input_details.name
            input_shape = self.input_details.shape
            input_type = self.input_details.type

            # Get output details
            self.output_name = self.session.get_outputs()[0].name

            print("--- ONNX Model Details ---")
            print(f"Input Name: {self.input_name}")
            print(f"Input Shape: {input_shape} (Note: 'None' or 'N' is the batch size)")
            print(f"Input Type: {input_type}")
            print("--------------------------")

        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            self.session = None
    def predict(self, preprocessed_image):
        try:
            # --- DIAGNOSTIC STEP ---
            # Print the shape and type of the input just before running the model
            print(f"DEBUG: Input tensor shape: {preprocessed_image.shape}, dtype: {preprocessed_image.dtype}")

            ort_inputs = {self.input_name: preprocessed_image}
            ort_outs = self.session.run([self.output_name], ort_inputs)
            
            logits = ort_outs[0]
            predicted_class_index = np.argmax(logits, axis=1)[0]
            
            return int(predicted_class_index) # Ensure it returns a standard Python int

        except Exception as e:
            print(f"!!!!!!!!!! ERROR during ONNX inference: {e} !!!!!!!!!!!")
            return None # Return None on failure