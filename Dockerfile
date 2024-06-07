# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make ports 8000 and 8001 available to the world outside this container
EXPOSE 8000
EXPOSE 8001

# Run main.py when the container launches
CMD ["uvicorn", "vit_serve:app", "--host", "0.0.0.0", "--port", "8000"]
