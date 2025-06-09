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
    """
    Loads an ONNX model and provides a method to run predictions.
    """
    def __init__(self, model_path="model.onnx"):
        print("Initializing ONNX Runtime session...")
        # For GPU, specify 'CUDAExecutionProvider'
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX model loaded. Input name: '{self.input_name}'.")

    def predict(self, preprocessed_image):
        """
        Runs the preprocessed image through the ONNX model and returns the 
        final predicted class index.
        """
        # The input to the session must be a dictionary
        ort_inputs = {self.input_name: preprocessed_image}
        
        # ort_outs is a list of numpy arrays, one for each model output
        ort_outs = self.session.run(None, ort_inputs)
        
        # 1. Get the first (and only) output array, which contains the logits
        logits = ort_outs[0]
        
        # 2. Find the index of the highest score along the class dimension (axis=1)
        #    This is the equivalent of torch.argmax()
        predicted_class_index = np.argmax(logits, axis=1)[0]
        
        return predicted_class_index
