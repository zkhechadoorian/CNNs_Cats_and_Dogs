# Start the FastAPI server using Uvicorn in the background
# - 'app:app' refers to the 'app' object in the 'app.py' file
# - '--host 0.0.0.0' makes the server accessible externally
# - '--port 5000' sets the port to 5000
# - '--workers 2' runs 2 worker processes for handling requests
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 2 & 

# Start the Streamlit application
# - 'home.py' is the Streamlit script to run
# - '--server.address 0.0.0.0' makes the app accessible externally
# - '--server.port 5001' sets the port to 5001
streamlit run home.py --server.address 0.0.0.0 --server.port 5001
