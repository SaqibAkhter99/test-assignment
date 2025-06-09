# test_server.py
import requests
import base64
import time
import argparse
import os
import numpy as np

# Constants for the Cerebrium API
CEREBRIUM_API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWVkNmY4MDY1IiwiaWF0IjoxNzQ5NDU3NDE1LCJleHAiOjIwNjUwMzM0MTV9.BqjfaGzQBHVGlOfL-3W_PXkOicLCsufz9J94XUWqmCOn_JY_oZC2MXJVetccQW7hPdFoYyrgKzFTarNsefMO4W7D-NoekdckdnzxE6xFH9zUVmuRJzhkeyUS_tywzopULmpyawXgObbYBfkX2mSZf6xta0n_ubVY7IaTnzHHuct-GZqcwVU3oNNyX8X3hk1OWAx7aG2l-MqpEbIKGL8okbopeYRVjo2gxRB4IDFQvcFnmYMLMp3xPkXYhH_9T8hbs1tJIVKrD1iVcRkX6sZQDNJFC4UUTBVMPJ64ctkpsbMCjdqVq6pa1WPbGunY3P_0Myk4LyM1rfYXxt38jb_-Uw"
ENDPOINT_URL = "https://api.cortex.cerebrium.ai/v4/p-ed6f8065/image-classifier-prod/run"



# test_server.py (CORRECTED VERSION)

def test_single_image(image_path: str):
    """Sends one image to the deployed endpoint and prints the result."""
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return None

    # Read the image file in binary mode and encode it to base64
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Prepare the JSON payload in the format the server expects
    payload = {"image_b64": image_b64}

    # Prepare the headers for authentication
    headers = {
        "Authorization": f"Bearer {CEREBRIUM_API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"\n--- Testing with image: {os.path.basename(image_path)} ---")
    
    start_time = time.time()
    response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    print(f"-> HTTP Status Code: {response.status_code}")
    print(f"-> Response Time: {latency_ms:.2f} ms")

    if response.status_code == 200:
        result = response.json()
        if "error" in result:
            print(f"-> Server-side error reported: {result['error']}")
            return None
        predicted_id = result.get("predicted_class_id")
        print(f"-> API Response: Predicted Class ID = {predicted_id}")
        return predicted_id
    else:
        print(f"-> HTTP Error: {response.text}")
        return None



def run_preset_test_suite():
    """Runs a series of tests with known outcomes against the deployed model."""
    print("\n--- Running Preset Test Suite on Deployed Model ---")
    tests = {
        "n01440764_tench.jpeg": 0,
        "n01667114_mud_turtle.jpeg": 35
    }
    passed_count = 0
    for image_file, expected_id in tests.items():
        predicted_id = test_single_image(image_file)
        if predicted_id == expected_id:
            print(f"✅ PASSED: Correctly classified '{image_file}'.")
            passed_count += 1
        else:
            print(f"❌ FAILED: Incorrectly classified '{image_file}'. Expected {expected_id}, got {predicted_id}.")
    
    print(f"\n--- Test Suite Summary: {passed_count}/{len(tests)} tests passed. ---")
    if passed_count != len(tests):
        exit(1) # Exit with error code for CI/CD integration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the deployed image classification model on Cerebrium.")
    parser.add_argument("--image", type=str, help="Path to a single image file to test.")
    parser.add_argument("--run-suite", action="store_true", help="Run the full preset test suite.")
    
    args = parser.parse_args()

    if CEREBRIUM_API_KEY == "YOUR_CEREBRIUM_API_KEY_HERE" or ENDPOINT_URL == "YOUR_ENDPOINT_URL_HERE":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Please update CEREBRIUM_API_KEY and ENDPOINT_URL in test_server.py !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit(1)

    if args.image:
        test_single_image(args.image)
    elif args.run_suite:
        run_preset_test_suite()
    else:
        print("Usage: python test_server.py --image <path_to_image> OR --run-suite")


