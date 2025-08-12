# Use the NVIDIA GPU base image with Python 3.12.8
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Add a build argument for the proxy
ARG http_proxy
ARG https_proxy

# Set proxy environment variables
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy

# Install Python and essential dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python3.12 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app
COPY requirements.txt .


# Install dependencies
RUN pip install -r requirements.txt
COPY . .


# Set an entrypoint or default command (optional)
# CMD ["python", "app.py"]  # Replace with your application's entry point
