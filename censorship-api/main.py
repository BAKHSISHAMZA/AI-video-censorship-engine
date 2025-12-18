import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Censorship API (ONNX)", version="2.0")

# 1. Load ONNX Session (CPU for wide compatibility)
# In production, we'd use 'CUDAExecutionProvider' if GPU is available
print("🚀 Loading ONNX Model...")
session = ort.InferenceSession("best_fixed.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1, 3, 640, 640]
output_names = [x.name for x in session.get_outputs()]
print("✅ Model Loaded. Input:", input_name)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # 1. Read Image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        original_shape = img.shape[:2]  # H, W

        # 2. Preprocessing (The Math: Normalize 0-1, Resize, Transpose)
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_input = img_norm.transpose(2, 0, 1)  # HWC -> CHW
        img_input = np.expand_dims(img_input, axis=0) # Add Batch Dim

        # 3. Inference (No PyTorch overhead)
        outputs = session.run(output_names, {input_name: img_input})
        raw_output = outputs[0]  # [1, 300, 4+cls]

        # 4. Post-Processing (Parsing the Tensor)
        detections = []
        # Access first batch
        batch_pred = raw_output[0] 
        boxes = batch_pred[:, :4] # cx, cy, w, h
        scores = batch_pred[:, 4:] # classes confidence
        
        # Get Max Score per box
        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        # Filter by Confidence > 0.4
        mask = max_scores > 0.4
        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_ids = class_ids[mask]

        for i in range(len(filtered_scores)):
            # Scale back to original image size
            cx, cy, w, h = filtered_boxes[i]
            cx *= original_shape[1] # width
            cy *= original_shape[0] # height
            w *= original_shape[1]
            h *= original_shape[0]
            
            detections.append({
                "class_id": int(filtered_ids[i]),
                "confidence": float(filtered_scores[i]),
                "bbox": [int(cx), int(cy), int(w), int(h)]
            })

        return {"detections": detections, "count": len(detections)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)