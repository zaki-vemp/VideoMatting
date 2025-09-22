# web_rvm_optimized.py
import os
import cv2
import torch
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, Response
from torchvision.transforms import ToTensor
from model import MattingNetwork
import threading
import time
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Config - Optimized settings
MODEL_PATH = os.path.join("checkpoint", "rvm_resnet50.pth")
DOWNSAMPLE_RATIO = 0.5  # Increased for better performance (was 0.25)
USE_CUDA = torch.cuda.is_available()
TARGET_WIDTH = 640  # Limit input resolution for better performance

# Global variables
model = None
device = None
to_tensor = ToTensor()
background_color = [0.47, 1.0, 0.6]  # Global background color

def load_model():
    global model, device
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f"Loading model on device: {device}")
    
    # Enable optimizations
    if USE_CUDA:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    model = MattingNetwork('resnet50').to(device).eval()
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt)
    
    # Enable half precision for faster inference on GPU
    if USE_CUDA and torch.cuda.get_device_capability()[0] >= 7:  # For RTX cards
        model = model.half()
        print("Using half precision (FP16) for faster inference")
    
    print("Model loaded successfully")

def preprocess_frame_gpu(frame_bgr, target_width=TARGET_WIDTH):
    """Optimized preprocessing with GPU acceleration"""
    # Resize frame first to reduce computation
    if target_width:
        h, w = frame_bgr.shape[:2]
        if w > target_width:
            scale = target_width / float(w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and move to GPU in one step
    img = to_tensor(rgb).unsqueeze(0).to(device, non_blocking=True)
    
    # Use half precision if available
    if USE_CUDA and model.dtype == torch.float16:
        img = img.half()
    
    return img

def tensor_to_bgr_image_optimized(tensor):
    """Optimized tensor to image conversion"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Move to CPU and convert to numpy in one step
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def process_frame_optimized(frame, rec_states):
    """Optimized frame processing with GPU acceleration"""
    global model, device, background_color
    
    if model is None:
        return frame, rec_states
    
    try:
        # Preprocess with GPU optimization
        src = preprocess_frame_gpu(frame)
        
        # Prepare background tensor
        bg_tensor = torch.tensor(background_color, device=device, dtype=src.dtype).view(3, 1, 1)
        
        with torch.no_grad():
            # Use torch.cuda.amp for automatic mixed precision if available
            if USE_CUDA:
                with torch.cuda.amp.autocast():
                    fgr, pha, *new_rec = model(src, *rec_states, DOWNSAMPLE_RATIO)
                    com = fgr * pha + bg_tensor * (1 - pha)
            else:
                fgr, pha, *new_rec = model(src, *rec_states, DOWNSAMPLE_RATIO)
                com = fgr * pha + bg_tensor * (1 - pha)
            
            # Convert back to image
            out_bgr = tensor_to_bgr_image_optimized(com.clamp(0, 1))
        
        return out_bgr, new_rec
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, rec_states

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Read image
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'error': 'Invalid image format'}), 400
    
    # Process image
    rec_states = [None] * 4
    processed_frame, _ = process_frame_optimized(frame, rec_states)
    
    # Convert back to base64
    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'processed_image': f'data:image/jpeg;base64,{img_base64}'
    })

@app.route('/webcam_stream')
def webcam_stream():
    """Optimized webcam stream with GPU acceleration"""
    def generate():
        cap = cv2.VideoCapture(0)
        
        # Optimize webcam settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(TARGET_WIDTH * 3/4))  # 4:3 aspect ratio
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
        
        rec_states = [None] * 4
        frame_count = 0
        start_time = time.time()
        
        # JPEG encoding parameters for better performance
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with GPU optimization
                processed_frame, rec_states = process_frame_optimized(frame, rec_states)
                
                # Encode frame with optimized settings
                _, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Performance monitoring
                frame_count += 1
                if frame_count % 30 == 0:  # Print FPS every 30 frames
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Current FPS: {fps:.2f}")
                
                # Remove sleep to maximize FPS
                # time.sleep(0.033)  # Removed this line
                
        finally:
            cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_background', methods=['POST'])
def change_background():
    """Change background color"""
    global background_color
    data = request.get_json()
    background_color = data.get('color', [0.47, 1.0, 0.6])
    
    return jsonify({'success': True, 'color': background_color})

@app.route('/get_device_info')
def get_device_info():
    """Get device information for debugging"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': str(device),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)
        info['cuda_memory_cached'] = torch.cuda.memory_reserved(0)
    
    return jsonify(info)

if __name__ == '__main__':
    print("Checking GPU availability...")
    if USE_CUDA:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU detected, using CPU")
    
    print("Loading RVM model...")
    load_model()
    print("Starting optimized Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)  # debug=False for better performance