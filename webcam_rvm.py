# webcam_rvm.py
import os
import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor
from model import MattingNetwork  # from the repo

# Config
MODEL_PATH = os.path.join("checkpoint", "rvm_resnet50.pth")
DOWNSAMPLE_RATIO = 0.25  # adjust if you want faster/slower
USE_CUDA = torch.cuda.is_available()
VIDEO_DEVICE = 0  # default webcam
TARGET_WIDTH = None  # leave None to use webcam native; you can set 640, 1280, etc.

# Helpers
to_tensor = ToTensor()

def preprocess_frame(frame_bgr, target_width=None):
    # convert BGR->RGB and normalize to [0,1]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if target_width is not None:
        h, w = rgb.shape[:2]
        scale = target_width / float(w)
        rgb = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    img = to_tensor(rgb).unsqueeze(0)  # 1x3xHxW, float in [0,1]
    return img

def tensor_to_bgr_image(tensor):
    # tensor: 3xHxW or 1x3xHxW
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1,2,0))
    arr = np.clip(arr*255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def main():
    # Load model
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print("Using device:", device)
    model = MattingNetwork('resnet50').to(device).eval()
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt)
    print("Model loaded:", MODEL_PATH)

    # Open webcam
    cap = cv2.VideoCapture(VIDEO_DEVICE)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # Optional: set capture width
    if TARGET_WIDTH:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)

    rec = [None] * 4  # initial recurrent states
    bgr_background = torch.tensor([.47, 1, .6]).view(3, 1, 1).to(device)  # green background normalized

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            src = preprocess_frame(frame, target_width=TARGET_WIDTH).to(device)  # 1x3xHxW
            with torch.no_grad():
                # model expects batch dimension for a chunk; we pass single frame
                # returns fgr, pha, *rec
                fgr, pha, *rec = model(src, *rec, DOWNSAMPLE_RATIO)
                # composite: fgr * pha + bgr * (1 - pha)
                com = fgr * pha + bgr_background * (1 - pha)
                out_bgr = tensor_to_bgr_image(com.clamp(0,1))
            # show
            cv2.imshow("RVM webcam (press q to quit)", out_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
