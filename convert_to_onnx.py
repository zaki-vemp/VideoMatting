# convert_to_onnx_fixed.py
import torch
import torch.onnx
import os
from model import MattingNetwork

class RVMWrapper(torch.nn.Module):
    """Wrapper to make RVM ONNX-compatible"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, src, r1i, r2i, r3i, r4i):
        # Fixed downsample ratio to avoid dynamic operations
        return self.model(src, r1i, r2i, r3i, r4i, 0.25)

def convert_rvm_to_onnx():
    print("Converting RVM PyTorch model to ONNX (fixed version)...")
    
    # Load the PyTorch model
    device = torch.device("cpu")
    base_model = MattingNetwork('resnet50').to(device).eval()
    
    # Load checkpoint
    MODEL_PATH = os.path.join("checkpoint", "rvm_resnet50.pth")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    base_model.load_state_dict(ckpt)
    
    # Wrap the model to fix downsample_ratio
    model = RVMWrapper(base_model)
    
    # Create dummy inputs with fixed sizes
    batch_size = 1
    height, width = 256, 256  # Fixed size for web
    
    src = torch.randn(batch_size, 3, height, width)
    r1i = torch.randn(batch_size, 16, height//4, width//4) 
    r2i = torch.randn(batch_size, 20, height//8, width//8)
    r3i = torch.randn(batch_size, 40, height//16, width//16)
    r4i = torch.randn(batch_size, 64, height//32, width//32)
    
    # Create output directory
    os.makedirs("web_model", exist_ok=True)
    
    # Export to ONNX with fixed parameters
    try:
        torch.onnx.export(
            model,
            (src, r1i, r2i, r3i, r4i),
            "web_model/rvm_resnet50.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['src', 'r1i', 'r2i', 'r3i', 'r4i'],
            output_names=['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o'],
            dynamic_axes={
                'src': {2: 'height', 3: 'width'},
                'fgr': {2: 'height', 3: 'width'},
                'pha': {2: 'height', 3: 'width'},
            }
        )
        print("✅ Model successfully exported to ONNX!")
        print("📁 Saved to: web_model/rvm_resnet50.onnx")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("The model is too complex for ONNX conversion.")
        print("Recommend using Solution 2 or 3 instead.")

if __name__ == "__main__":
    convert_rvm_to_onnx()