import numpy as np
from PIL import Image
import onnxruntime as ort
import os
class ImagePreprocessor:
    """
    Handles all pre-processing steps required to transform an input image
    into a tensor suitable for the classification model.
    """
    def __init__(self, size=(224, 224)):
        self.size = size
        # Normalization values for ImageNet in [R, G, B] order
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image_input):
        """
        Takes a file path or PIL Image, performs all necessary pre-processing,
        and returns a numpy tensor with the correct shape and type.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError("Input must be a file path string or a PIL Image object.")

        img = img.convert("RGB")
        
        # Use Image.Resampling.BILINEAR for modern Pillow versions
        img = img.resize(self.size, Image.Resampling.BILINEAR)

        img_np = np.array(img, dtype=np.float32) / 255.0
        normalized_img = (img_np - self.mean) / self.std

        # --- THIS IS A CRITICAL FIX ---
        # Transpose from [Height, Width, Channels] to [Channels, Height, Width]
        # The model expects shape (N, C, H, W) but np.array(img) gives (H, W, C)
        transposed_img = normalized_img.transpose((2, 0, 1))

        # Add a batch dimension to create a [1, C, H, W] tensor
        return np.expand_dims(transposed_img, axis=0).astype(np.float32)

class OnnxModel:
    """
    Loads an ONNX model and provides a method to run predictions.
    """
    def __init__(self, model_path="model.onnx"):
        self.session = None
        self.init_error = None
        try:
            script_dir = os.path.dirname(__file__)
            absolute_model_path = os.path.join(script_dir, model_path)
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(absolute_model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            self.init_error = str(e)

    def predict(self, preprocessed_image):
        """
        Runs inference and returns the raw array of output scores (logits).
        """
        if not self.session:
            raise RuntimeError(f"MODEL FAILED TO LOAD: {self.init_error}")
            
        try:
            ort_inputs = {self.input_name: preprocessed_image}
            ort_outs = self.session.run([self.output_name], ort_inputs)
            
            # --- THIS IS THE FIX ---
            # Return the raw output array from the ONNX model.
            # Do NOT calculate argmax here. Let the caller do it.
            return ort_outs[0] 
            
        except Exception as e:
            raise RuntimeError(f"INFERENCE FAILED: {str(e)}")