
# Use a base image with Python
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Hugging Face cache directory to a writable location
ENV HF_HOME=/tmp/huggingface
RUN mkdir -p /tmp/huggingface/hub /tmp/huggingface/transformers \
 && chmod -R 777 /tmp/huggingface   # <-- key change

# Copy the hosting script
COPY app.py /app/app.py

# Hugging Face Spaces expect apps to listen on port 7860
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

