## Use official Python slim image as base
FROM python:3.10-slim

## Install AWS CLI and update system packages
RUN apt update -y && apt install -y awscli

## Set working directory
WORKDIR /app


## Copy all project files into the container
COPY . .

## Install Python dependencies
RUN pip install -r requirements.txt

## Make the startup script executable
RUN chmod +x start_uvicorn_streamlit.sh

## Expose the application port
EXPOSE 5001

## Start the application using the startup script
CMD [ "bash","start_uvicorn_streamlit.sh" ]