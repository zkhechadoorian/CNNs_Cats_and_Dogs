# Import necessary FastAPI modules and other dependencies
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import os
from ImageClassification.utils import decodeImage
# Our pipeline.
from ImageClassification.pipeline.predict import DogCat
import uvicorn
from pydantic import BaseModel, Field

# Set environment variables for language and locale
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize FastAPI app
app = FastAPI(docs_url="/docs", redoc_url="/redoc")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"],         # Allow all headers
)

# Set up Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# ClientApp class to manage image filename and classifier instance
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"         # Default filename for input image
        self.classifier = DogCat(self.filename)  # Initialize classifier with filename

# Create a global instance of ClientApp
clApp = ClientApp()

# Route for home page, renders index.html template
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to trigger training by running main.py script
@app.post("/train")
async def trainRoute():
    os.system("python main.py")  # Execute training script
    return {"message": "Training done successfully!"}

# Route to handle image prediction requests
class ImageRequest(BaseModel):
    image: str = Field(description="Base64 encoded image string")

@app.post("/predict")
async def predictRoute(image_request: ImageRequest):
    """Handle image prediction requests, must be passed as base64"""
    image = image_request.image                        # Get image data from pydantic model
    decodeImage(image, clApp.filename)                 # Decode and save image to file
    result = clApp.classifier.predictiondogcat()       # Run prediction on saved image
    return JSONResponse(content=result)                # Return prediction result as JSON

if __name__ == "__main__":
    uvicorn.run(app,port=5000, workers=2)  # Run FastAPI app with Uvicorn