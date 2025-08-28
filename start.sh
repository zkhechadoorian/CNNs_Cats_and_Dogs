# Build a Docker image from the current directory and tag it as 'img-class-transfer'
docker build -t img-class-transfer .

# Stop the running container named 'img-class-transfer' (if it exists) and remove it
docker stop img-class-transfer && docker rm img-class-transfer

# Run a new container in detached mode with the name 'img-class-transfer'
# Map host ports 5000 and 5001 to the container's ports 5000 and 5001
# Always restart the container unless explicitly stopped
docker run -d --name img-class-transfer -p 5000:5000 -p 5001:5001 --restart always img-class-transfer