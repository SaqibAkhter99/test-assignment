# Production Image Classification Model on Cerebrium

This project demonstrates the end-to-end process of deploying a PyTorch-based image classification model to a production-grade, serverless GPU environment on Cerebrium. The process includes model optimization via ONNX conversion, containerization with Docker, and rigorous local and remote testing.

## Project Structure

```
.
├── Dockerfile              # Defines the container environment for deployment.
├── README.md               # This documentation file.
├── cerebrium.toml          # Cerebrium deployment configuration.
├── convert_to_onnx.py      # Script to convert the PyTorch model to ONNX.
├── main.py                 # Server entrypoint for Cerebrium.
├── model.py                # Contains pre-processing and ONNX inference logic.
├── pytorch_model.py        # The original PyTorch model definition (ResNet).
├── requirements.txt        # Python dependencies.
├── test.py                 # Local test suite for the model and pipeline.
├── test_server.py          # Test suite for the live deployed endpoint.
├── n01440764_tench.jpeg    # Sample image for class 0.
├── n01667114_mud_turtle.jpeg # Sample image for class 35.
└── weights/
    └── pytorch_model_weights.pth # Model weights (must be downloaded).
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- An account on [Cerebrium](https://www.cerebrium.ai/).
- Cerebrium CLI installed (`pip install cerebrium`) and configured (`cerebrium login`).

### Step-by-Step Instructions

1.  **Clone the Repository:**
    ```
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install Dependencies:**
    ```
    pip install -r requirements.txt
    ```

3.  **Download Model Weights:**
    Download the model weights from this link: `https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1`
    
    Create a `weights` directory and place the downloaded file inside it. Rename the file to `pytorch_model_weights.pth`.
    ```
    mkdir weights
    mv /path/to/downloaded/file.pth weights/pytorch_model_weights.pth
    ```

## Usage

### 1. Convert Model to ONNX
This step optimizes the model for fast inference. The output `model.onnx` will be created in the root directory.
```
python convert_to_onnx.py
```

### 2. Run Local Tests
Before deploying, verify that the entire pipeline works correctly on your local machine. This test compares the ONNX model's output to the original PyTorch model.
```
python test.py
```

### 3. Deploy to Cerebrium
Deploy the application using the Cerebrium CLI. This will use the `Dockerfile` to build and deploy your container.
```
cerebrium deploy
```
After a successful deployment, the CLI will provide you with an API endpoint URL and an authentication token. Update these in `test_server.py`.

### 4. Test the Deployed Endpoint
You can test the live server in two ways:

**A. Test a single image:**
```
python test_server.py --image_path n01440764_tench.jpeg
```

**B. Run the full test suite:**
This will test the two sample images and verify their class IDs.
```
python test_server.py --test-suite
```