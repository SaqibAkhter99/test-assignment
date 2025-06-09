import os
import numpy as np
from PIL import Image
import onnxruntime as ort

class Model:
    """
    A single, self-contained class for the image classification model.
    This structure is required for deployment on platforms like Cerebrium.
    """
    def __init__(self):
    # --- Preprocessor Initialization ---
        self.size = (224, 224)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # --- ONNX Model Initialization (NO try...except block) ---
        # This will ensure the real error is shown in the server logs.
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "model.onnx")
        
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name


    def preprocess(self, image_input):
        """
        Performs all pre-processing steps on a PIL Image.
        """
        img = image_input.convert("RGB")
        img = img.resize(self.size, Image.Resampling.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        normalized_img = (img_np - self.mean) / self.std
        transposed_img = normalized_img.transpose((2, 0, 1))
        return np.expand_dims(transposed_img, axis=0).astype(np.float32)

    def predict(self, item):
        """
        The main prediction entry point called by the server.
        It handles the full pipeline: preprocessing -> inference.
        """
        # Check if the model failed to load during initialization
        if not self.session:
            raise RuntimeError(f"MODEL FAILED TO LOAD: {self.init_error}")
            
        try:
            # The server provides the raw PIL image in the 'item' dictionary
            image = item["image"]
            
            # Use the instance's own preprocessor method
            preprocessed_image = self.preprocess(image)
            
            # Run inference using the instance's ONNX session
            ort_inputs = {self.input_name: preprocessed_image}
            logits = self.session.run([self.output_name], ort_inputs)[0]
            
            predicted_id = np.argmax(logits, axis=1)[0]
            
            # Return the final class ID as a simple integer
            return int(predicted_id)
            
        except Exception as e:
            # This will catch any errors during the prediction pipeline
            raise RuntimeError(f"INFERENCE FAILED: {str(e)}")

