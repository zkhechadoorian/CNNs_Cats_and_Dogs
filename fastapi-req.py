import requests
import base64

# Path to your image file
image_path = "10493.jpg"


server_ip = "127.0.0.1"
# FastAPI endpoint URL
url = f"http://{server_ip}:8000/predict"

# Read and encode image to base64
with open(image_path, "rb") as img_file:
    b64_image = base64.b64encode(img_file.read()).decode("utf-8")

# Prepare payload
payload = {"image": b64_image}



# Send POST request
response = requests.post(url, json=payload)

# Print response
print(response.status_code)
print(response.json())