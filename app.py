# This script launches a FastAPI web application for serving the cat vs. dog image classifier.
# It provides endpoints for home page rendering, model training, and image prediction.
# The app supports CORS, uses Jinja2 templates, and handles base64 image uploads for prediction.

# Import necessary FastAPI modules and other dependencies
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import os
from ImageClassification.utils import decodeImage
from ImageClassification.pipeline.predict import DogCat
import uvicorn
from pydantic import BaseModel, Field

# Set environment variables for language and locale
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize FastAPI app with custom documentation URLs
app = FastAPI(docs_url="/docs", redoc_url="/redoc")

# Add CORS middleware to allow cross-origin requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2 templates directory for HTML rendering
templates = Jinja2Templates(directory="templates")

class ClientApp:
    """
    Manages the input image filename and the classifier instance.
    """
    def __init__(self):
        self.filename = "inputImage.jpg"         # Default filename for input image
        self.classifier = DogCat(self.filename)  # Initialize classifier with filename

# Create a global instance of ClientApp
clApp = ClientApp()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the home page using the index.html Jinja2 template.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/train")
async def trainRoute():
    """
    Triggers model training by running main.py as a subprocess.
    """
    os.system("python main.py")
    return {"message": "Training done successfully!"}

class ImageRequest(BaseModel):
    """
    Pydantic model for image prediction requests.
    Expects a base64-encoded image string.
    """
    image: str = Field(description="Base64 encoded image string")

@app.post("/predict")
async def predictRoute(image_request: ImageRequest):
    """
    Handles image prediction requests.
    Decodes the base64 image, saves it, and returns the prediction result.
    """
    image = image_request.image
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predictiondogcat()
    return JSONResponse(content=result)

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn on port 5000 with 2 worker processes
    uvicorn.run(app, port=5000, workers=2)