import runpod
import subprocess
import time
import requests
import base64
from io import BytesIO

# Start the FastAPI application as a subprocess
def start_fastapi():
    subprocess.Popen(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"])
    
    # Wait for the server to start
    while True:
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("FastAPI server is running")
                break
        except:
            print("Waiting for FastAPI server to start...")
            time.sleep(1)

# Start the FastAPI server when the handler is imported
start_fastapi()

# Define the handler function
def handler(event):
    try:
        # Extract the input data
        input_data = event.get("input", {})
        
        # Check if we have the required fields
        if "user_image_base64" not in input_data or "garment_image_base64" not in input_data:
            return {"error": "Missing required fields: user_image_base64 and garment_image_base64"}
        
        # Forward the request to the FastAPI application
        response = requests.post(
            "http://localhost:8000/try-on/",
            json={
                "user_image_base64": input_data["user_image_base64"],
                "garment_image_base64": input_data["garment_image_base64"]
            }
        )
        
        # Return the response from the FastAPI application
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"FastAPI returned status code {response.status_code}: {response.text}"}
    
    except Exception as e:
        return {"error": str(e)}

# Start the runpod handler
runpod.serverless.start({"handler": handler})
