# Use a base image with Python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /Cityscapes_Moustafa

# Copy the requirements.txt file into the container
COPY requirements.txt ./

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /Cityscapes_Moustafa

# Expose port if needed (depends on how you want to run your script)
EXPOSE 8080

# Define the command to run your application
CMD ["python", "Evaluation.py"]

