import torch
import numpy as np
import onnxruntime as ort
import cv2
from ultralytics import RTDETR
import argparse
import sys
from pathlib import Path

def export_and_debug(model_path, output_path):
    print(f"Loading model: {model_path}")
    model = RTDETR(model_path)
    
    # --- STEP 1: EXPORT STRATEGY ---
    # We use opset=17 (Crucial for Transformers)
    # We DISABLE simplify initially to rule out graph corruption
    # We use dynamic=True, but if this fails, we will fallback to static later
    print("\n--- STARTING EXPORT ---")
    try:
        success = model.export(
            format="onnx",
            opset=17,           # CRITICAL for RT-DETR
            simplify=False,     # CRITICAL: Simplifier often breaks RT-DETR attention heads
            dynamic=True,       # Allow dynamic batching
            imgsz=640           # Standard size
        )
        print(f"Export command returned: {success}")
    except Exception as e:
        print(f"Export failed: {e}")
        return

    # Ultralytics auto-names the file. Let's find it.
    exported_file = Path(model_path).with_suffix('.onnx')
    if str(exported_file) != output_path:
        import shutil
        shutil.move(exported_file, output_path)
    
    print(f"Model saved to: {output_path}")

    # --- STEP 2: RAW TENSOR VERIFICATION ---
    print("\n--- RAW TENSOR DIAGNOSTICS ---")
    print("Running inference on dummy data to check raw values...")
    
    # Create a dummy image (Random noise)
    # Setup exactly as Ultralytics expects: [1, 3, 640, 640], RGB, 0-1 Normalized
    dummy_img = np.random.uniform(0, 1.0, (1, 3, 640, 640)).astype(np.float32)
    
    sess_options = ort.SessionOptions()
    # Disable optimizations for debugging
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    session = ort.InferenceSession(output_path, sess_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_names = [x.name for x in session.get_outputs()]
    
    print(f"Input Name: {input_name}")
    print(f"Output Names: {output_names}")
    
    # Run Inference
    outputs = session.run(output_names, {input_name: dummy_img})
    raw_output = outputs[0] # Usually [batch, 300, classes + 4] or [batch, 300, 4 + classes]
    
    print(f"Output Shape: {raw_output.shape}")
    
    # --- STEP 3: ANALYZE OUTPUTS ---
    # RT-DETR output from Ultralytics is typically: [Batch, 300, 4bbox + n_classes]
    # or sometimes split into two outputs depending on version.
    # Assuming combined:
    
    batch_idx = 0
    # Slice the tensor
    # If shape is [1, 300, 84] (80 classes + 4 coords)
    num_queries = raw_output.shape[1]
    last_dim = raw_output.shape[2]
    
    # Attempt to identify Box vs Class channels
    # Usually boxes are first 4 or last 4. 
    # Ultralytics RT-DETR default: [x, y, w, h, class1, class2, ...]
    
    boxes = raw_output[batch_idx, :, :4]
    scores = raw_output[batch_idx, :, 4:]
    
    max_score = np.max(scores)
    avg_score = np.mean(scores)
    max_box_coord = np.max(boxes)
    
    print("\n--- DIAGNOSTIC RESULTS ---")
    print(f"Max Confidence Score found in tensor: {max_score:.6f}")
    print(f"Avg Confidence Score: {avg_score:.6f}")
    print(f"Max Box Coordinate value: {max_box_coord:.2f}")
    
    if max_score == 0.0:
        print("\n❌ CRITICAL ERROR: Model is outputting pure zeros.")
        print("Possible causes:")
        print("1. Opset version < 17 (We used 17, so likely not this)")
        print("2. Input normalization is wrong (We used 0-1 float)")
        print("3. The model weights were lost during export")
    elif max_score < 0.1:
        print("\n⚠️ WARNING: Model is running, but confidences are extremely low.")
        print("This is normal for random noise inputs, but check with a real image.")
    else:
        print("\n✅ SUCCESS: Model is producing valid non-zero outputs.")
        print("The ONNX file is technically sound. If detections are missing in your app,")
        print("check your post-processing (NMS, Thresholding) code.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .pt model')
    parser.add_argument('--output', type=str, default='rtdetr_fixed.onnx', help='Path to output .onnx')
    args = parser.parse_args()
    
    export_and_debug(args.model, args.output)