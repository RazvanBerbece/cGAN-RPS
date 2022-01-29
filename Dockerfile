# syntax=docker/dockerfile:1

# Pick Docker image with GPU support so Tensorflow can be used
FROM ubuntu:18.04

MAINTAINER "Antonio Berbece"

# Install default libs & python3
RUN apt-get update && apt-get install -y    \
    python3-pip &&                          \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

# Define the << working directory >> in which RUN, CMD, ADD, COPY, etc. are run
WORKDIR /root

# Install app dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --quiet --no-cache-dir -r requirements.txt

# Copy project files (libs, resources, models, etc.) into container (except for files included in .dockerignore)
COPY . .

# Run script with server entrypoint 
CMD ["python3", "server/app.py"]
