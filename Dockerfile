
# Use a base image with Python
FROM python:3.10-slim

# Install system deps if needed (optional but safe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face writable cache on Spaces
ENV HF_HOME=/tmp/huggingface \
    HF_HUB_CACHE=/tmp/huggingface/hub

# Copy app
COPY app.py /app/app.py

# Spaces healthcheck expects 7860
EXPOSE 7860

# Run Streamlit on the right port/interface
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]



# # Use a base image with Python
# FROM python:3.9-slim

# # Set the working directory
# WORKDIR /app

# # Copy the requirements file and install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the hosting script and model
# COPY app.py .
# #COPY model /app/model

# # Expose the port
# EXPOSE 8501

# # Run the Streamlit application
# CMD ["streamlit", "run", "app.py"]

