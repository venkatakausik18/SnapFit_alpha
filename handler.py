import runpod
import subprocess
import time
import requests
import os
import signal
import threading
import base64

# Global variable to track if the FastAPI app is running
fastapi_process = None
fastapi_port = 8000

def start_fastapi_app():
    """Start the FastAPI application as a subprocess"""
    global fastapi_process
    if fastapi_process is None:
        # Start the FastAPI app using uvicorn
        fastapi_process = subprocess.Popen(
            ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(fastapi_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Wait for the FastAPI app to start (models will load during this time)
        print("Starting FastAPI application and loading models...")
        time.sleep(240)  # Give more time for model loading
        print("FastAPI application started on port", fastapi_port)

def stop_fastapi_app():
    """Stop the FastAPI application subprocess"""
    global fastapi_process
    if fastapi_process is not None:
        fastapi_process.terminate()
        fastapi_process.wait()
        fastapi_process = None
        print("FastAPI application stopped")

def handler(job):
    """
    Handler function for RunPod serverless.
    This forwards requests to your FastAPI application.
    """
    # Start the FastAPI app if it's not already running
    start_fastapi_app()
    
    # Get the input from the job
    job_input = job["input"]
    
    # Extract the base64 images from the input
    user_image_base64 = job_input.get("user_image_base64", "")
    garment_image_base64 = job_input.get("garment_image_base64", "")
    
    if not user_image_base64 or not garment_image_base64:
        return {"error": "Both user_image_base64 and garment_image_base64 are required"}
    
    # Prepare the payload for the FastAPI app
    payload = {
        "user_image_base64": user_image_base64,
        "garment_image_base64": garment_image_base64
    }
    
    try:
        # Forward the request to your FastAPI app's try-on endpoint
        url = f"http://localhost:{fastapi_port}/try-on/"
        response = requests.post(url, json=payload, timeout=120)  # Longer timeout for image processing
        
        # Return the response
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"FastAPI returned status code {response.status_code}",
                "detail": response.text
            }
    except Exception as e:
        return {"error": str(e)}

# Register a cleanup function to stop the FastAPI app when the handler exits
def cleanup():
    stop_fastapi_app()

# Start the serverless function
runpod.serverless.start({"handler": handler, "cleanup_function": cleanup})
