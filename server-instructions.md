# Login to server 

SSH to your user and follow the rest of instructions here. 


# 🐍 Create Virtual Environment

```sh
python3 -m venv venv && source venv/bin/activate
```

# 📥 Install gdown Command Line

`gdown` allows us to download Google Drive files using their Drive ID.

```sh
pip install gdown
```

# 💻 Retrieve Our Code

To retrieve our code, run the following command:

```sh
gdown 1-SXET0LdA0IW6RcoO0_CcznV9-gcE6VB
```

This will download a zipped file called `Compu-Img-Class.zip`.

# 📦 Extracting Zipped File

To extract the zip file data into the folder `Compu-Img-Class`:

```sh
unzip Compu-Img-Class.zip -d Compu-Img-Class
```

# 🚀 Starting Our Server

The following command will start a bash script that runs a series of commands to start our Docker service:

```sh
cd Compu-Img-Class && bash start.sh
```
The explaination for these commands are below.

# 📜 Scripts & Docker Explanations

This section explains how the provided scripts and Docker setup work together to deploy your FastAPI and Streamlit applications.

## 🏁 start.sh

The `start.sh` script automates the Docker workflow:

```sh
docker build -t img-class-transfer .
docker stop img-class-transfer && docker rm img-class-transfer
docker run -d --name img-class-transfer -p 5000:5000 -p 5001:5001 --restart always img-class-transfer
```
- 🏗️ **Build** — Creates the Docker image from the `Dockerfile`.
- 🛑 **Stop & Remove** — Cleans up any existing container.
- 🚦 **Run** — Starts a new container, maps ports, and ensures auto-restart.

## 🐳 Dockerfile

The Dockerfile packages the application and its dependencies into a portable container:

```dockerfile
FROM python:3.10-slim

RUN apt update -y && apt install -y awscli
WORKDIR /app

COPY . .
RUN pip install -r requirements.txt
RUN chmod +x start_uvicorn_streamlit.sh
EXPOSE 5001

CMD [ "bash","start_uvicorn_streamlit.sh" ]
```
- Uses `python:3.10-slim` as the base image.
- Installs AWS CLI (optional).
- Sets `/app` as the working directory.
- Copies all contents from the current directory.
- Installs Python requirements.
- Makes `start_uvicorn_streamlit.sh` executable.
- Exposes port 5001.
- Sets the container's start command.

## ⚡ start_uvicorn_streamlit.sh

This script launches both FastAPI and Streamlit apps, exposing ports `5000` and `5001`:

```sh
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 2 & 
streamlit run home.py --server.address 0.0.0.0 --server.port 5001
```
- FastAPI (Uvicorn) runs on port 5000.
- Streamlit runs on port 5001.
- Both services accept requests from any public IP address (`0.0.0.0`).

---

# 🌐 Accessing the Apps in Your Browser

Once the server is running, access the deployed applications:

- **⚡ FastAPI**  
    - Website: `http://<server-ip>:5000/`
    - API docs: `http://<server-ip>:5000/docs` (Swagger UI)

- **🎨 Streamlit App:**  
    - Web UI: `http://<server-ip>:5001`

> 🔄 Replace `<server-ip>` with your server's actual IP address or domain name.

## FastAPI API Endpoints Explaination

The FastAPI Swagger UI at `/docs` exposes three main endpoints:

- **`/` (Home):**  
    - `GET` — Serves the Jinja-based web application for predictions. It will return an html response that needs to be rendered by the browser.
- **`/train`:**  
    - `POST` — Triggers the model training pipeline.
- **`/predict`:**  
    - `POST` — Accepts images for classification via REST API.


To trigger these endpoints on your right there's a button called `try it out`, please press on it. After that another button will appear in a blue color called `Execute`, please press on it to try the endpoint.
---

✨ With these steps, your FastAPI and Streamlit apps are deployed and accessible online!


---

# 🐍 Invoking the Endpoints via Python

To demonstrate how to call the REST API, we'll use the `requests` package in Python to interact with the endpoints.

The REST API endpoint accepts image data as JSON in the following format:

```json
{
    "image": "b64_image_string"
}
```

Therefore, our script will look as follows at `fastapi-req.py` :

```python
import requests
import base64

# Path to your image file
image_path = "10493.jpg"


server_ip = "127.0.0.1"
# FastAPI endpoint URL
url = f"http://{server_ip}:5000/predict"

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
```

## 🖼️ Why do we encode our images to base64?

Encoding images into base64 is commonly done for these reasons:

- 📄 **Embedding in Text-Based Formats:**  
    Base64 allows binary image data to be represented as plain text, making it easy to embed images directly in HTML, CSS, JSON, or XML files.

- 🔗 **Data Transfer:**  
    It simplifies sending images over protocols that only support text (like email or some APIs).

- 📦 **Avoiding File Handling:**  
    Embedding images as base64 strings can reduce the need for separate image files, simplifying deployment and portability.
