import runpod
import subprocess
import time
import requests
import socket
import asyncio
import random

# Global variables
fastapi_process = None
fastapi_port = 8000
request_rate = 0

def is_fastapi_app_ready():
    """Check if the FastAPI app is running and ready to accept connections."""
    try:
        # Try to establish a connection to FastAPI on localhost:8000
        with socket.create_connection(("localhost", fastapi_port), timeout=5):
            return True
    except (socket.timeout, ConnectionRefusedError):
        return False

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
        print("Starting FastAPI application and loading models...")
        
        # Wait for the FastAPI app to start and be ready
        while not is_fastapi_app_ready():
            time.sleep(2)  # Check every 2 seconds if the FastAPI app is ready
        print("FastAPI application started on port", fastapi_port)

def stop_fastapi_app():
    """Stop the FastAPI application subprocess"""
    global fastapi_process
    if fastapi_process is not None:
        fastapi_process.terminate()
        fastapi_process.wait()
        fastapi_process = None
        print("FastAPI application stopped")

async def handler(job):
    """
    Asynchronous handler function for RunPod serverless.
    This forwards requests to your FastAPI application with concurrency support.
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
        
        # Wait for FastAPI to be ready before sending the request
        while not is_fastapi_app_ready():
            print("Waiting for FastAPI to be ready...")
            await asyncio.sleep(2)  # Async sleep to not block other requests

        # Send POST request to FastAPI
        # Using a synchronous request inside an async function
        # For better performance, consider using aiohttp for async HTTP requests
        response = requests.post(url, json=payload, timeout=180)  # Longer timeout for image processing
        
        # Return the response
        if response.status_code == 200:
            return response.json()
        else:
            print(f"FastAPI returned status code {response.status_code}: {response.text}")
            return {
                "error": f"FastAPI returned status code {response.status_code}",
                "detail": response.text
            }
    except Exception as e:
        print(f"Error while calling FastAPI: {e}")
        return {"error": str(e)}

def adjust_concurrency(current_concurrency):
    """
    Adjusts the concurrency level based on the current request rate.
    """
    global request_rate
    update_request_rate()  # Update the request rate

    max_concurrency = 10
    min_concurrency = 1
    high_request_rate_threshold = 50

    if (
        request_rate > high_request_rate_threshold
        and current_concurrency < max_concurrency
    ):
        return current_concurrency + 1
    elif (
        request_rate <= high_request_rate_threshold
        and current_concurrency > min_concurrency
    ):
        return current_concurrency - 1
    return current_concurrency

def update_request_rate():
    """
    Updates the request rate based on actual or simulated metrics.
    In a production environment, you might want to track actual request rates.
    """
    global request_rate
    request_rate = random.randint(20, 100)  # Simulate changes in request rate

# Register a cleanup function to stop the FastAPI app when the handler exits
def cleanup():
    stop_fastapi_app()

# Start the serverless function with concurrency support
runpod.serverless.start({
    "handler": handler, 
    "cleanup_function": cleanup,
    "concurrency_modifier": adjust_concurrency
})
