# Production Image Classification API on Cerebrium

This project demonstrates the end-to-end deployment of a production-ready image classification model on Cerebrium's serverless GPU platform. The implementation follows MLOps best practices, including model optimization (PyTorch to ONNX), containerization with Docker, and comprehensive local and remote testing.

## Project Deliverables

-   `convert_to_onnx.py`: Converts the PyTorch model to the efficient ONNX format.
-   `model.py`: Contains modular classes for image pre-processing and ONNX model inference.
-   `test.py`: A local test suite to validate the entire pipeline before deployment.
-   `cerebrium.toml`, `Dockerfile`, `main.py`: Configuration and code for Docker-based deployment on Cerebrium.
-   `test_server.py`: A client script to test the live, deployed API endpoint.
-   `README.md`: This setup and usage guide.

## Prerequisites

1.  Python 3.8+
2.  A Cerebrium account with the CLI installed: `pip install cerebrium`
3.  Git for version control.
4.  Docker installed and running on your local machine (for potential local builds, though not required for Cerebrium deployment).

## Step-by-Step Instructions

### 1. Initial Setup

**Clone the repository and navigate into the directory:**
```
git clone <your-repo-url>
cd <repo-name>
```

**Install Python dependencies:**
```
pip install -r requirements.txt
```

**Download model weights:**
Download the weights from [this link](https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1). Create a `weights` directory and move the downloaded file into it, ensuring it's named `pytorch_model_weights.pth`.
```
mkdir weights
# Example command to move the file
mv ~/Downloads/pytorch_model_weights.pth weights/
```

### 2. Local Preparation and Testing

**Convert the model to ONNX:**
This step creates the `model.onnx` file required for deployment.
```
python convert_to_onnx.py
```

**Run the local test suite:**
Verify that all components work correctly on your machine before deploying.
```
python test.py
```
You should see `--- All Local Tests Passed Successfully! ---` at the end.

### 3. Deployment to Cerebrium

**Log in to Cerebrium:**
Authenticate your machine with your Cerebrium account.
```
cerebrium login
```

**Deploy the application:**
From the project's root directory, execute the deploy command. Cerebrium will use the `Dockerfile` to build and deploy your containerized application.
```
cerebrium deploy
```

### 4. Testing the Deployed API

**Update `test_server.py`:**
After deployment, Cerebrium will provide you with an **API Key** and an **Endpoint URL**. Open `test_server.py` and replace the placeholder values at the top of the file with your credentials.

**Run tests against the live endpoint:**

*   **To test a single image:**
    ```
    python test_server.py --image n01440764_tench.jpeg
    ```

*   **To run the full test suite:**
    ```
    python test_server.py --run-suite
    ```
*  **To trigger the flow from postman:**
   ```
   curl --location 'https://api.cortex.cerebrium.ai/v4/p-ed6f8065/image-classifier-prod/run' \
   --header 'Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWVkNmY4MDY1IiwiaWF0IjoxNzQ5NDU3NDE1LCJleHAiOjIwNjUwMzM0MTV9.BqjfaGzQBHVGlOfL-3W_PXkOicLCsufz9J94XUWqmCOn_JY_oZC2MXJVetccQW7hPdFoYyrgKzFTarNsefMO4W7D-NoekdckdnzxE6xFH9zUVmuRJzhkeyUS_tywzopULmpyawXgObbYBfkX2mSZf6xta0n_ubVY7IaTnzHHuct-GZqcwVU3oNNyX8X3hk1OWAx7aG2l-MqpEbIKGL8okbopeYRVjo2gxRB4IDFQvcFnmYMLMp3xPkXYhH_9T8hbs1tJIVKrD1iVcRkX6sZQDNJFC4UUTBVMPJ64ctkpsbMCjdqVq6pa1WPbGunY3P_0Myk4LyM1rfYXxt38jb_-Uw' \
   --header 'Content-Type: application/json' \
   --data '{
      "item": {
         "image_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
      }
   }'
   ```
This will test the two sample images and provide a pass/fail summary, along with latency and status code monitoring for each request.

