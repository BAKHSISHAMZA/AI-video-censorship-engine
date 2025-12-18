# 1. Base Image: Pre-loaded with OpenCV, PyTorch, and System Libs
FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

# 2. Install ONLY the web server and ONNX (Everything else is already there)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    onnxruntime

# 3. Copy your files
# Make sure the ONNX filename matches what is in your folder!
# Based on your last screenshot, it is "best_fixed.onnx"
COPY main.py .
COPY best_fixed.onnx .

# 4. Run it
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]