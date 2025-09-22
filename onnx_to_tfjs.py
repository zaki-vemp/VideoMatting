import os
import subprocess
import sys

def convert_onnx_to_tfjs():
    """Convert ONNX model to TensorFlow.js format"""
    
    # Paths
    onnx_path = "web_model/rvm_resnet50_fp32.onnx"
    saved_model_path = "web_model/tf_saved_model"
    tfjs_path = "web_model/tfjs_model"
    
    print("🔄 Converting ONNX → TensorFlow SavedModel → TensorFlow.js")
    
    # Step 1: ONNX → TensorFlow SavedModel
    print("Step 1: Converting ONNX to TensorFlow SavedModel...")
    try:
        import onnx
        import tf2onnx
        import tensorflow as tf
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = tf2onnx.convert.from_onnx(onnx_model, output_path=saved_model_path)
        print("✅ Successfully converted to TensorFlow SavedModel")
        
    except Exception as e:
        print(f"❌ Error in Step 1: {e}")
        print("Trying alternative method...")
        
        # Alternative: Use command line
        cmd = f"python -m tf2onnx.convert --onnx {onnx_path} --saved-model {saved_model_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully converted to TensorFlow SavedModel (alternative method)")
        else:
            print(f"❌ Alternative method failed: {result.stderr}")
            return False
    
    # Step 2: TensorFlow SavedModel → TensorFlow.js
    print("Step 2: Converting TensorFlow SavedModel to TensorFlow.js...")
    try:
        cmd = f"tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default {saved_model_path} {tfjs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully converted to TensorFlow.js!")
            print(f"📁 TensorFlow.js model saved to: {tfjs_path}")
            return True
        else:
            print(f"❌ Error in Step 2: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error in Step 2: {e}")
        return False

if __name__ == "__main__":
    # Create directories
    os.makedirs("web_model/tf_saved_model", exist_ok=True)
    os.makedirs("web_model/tfjs_model", exist_ok=True)
    
    success = convert_onnx_to_tfjs()
    
    if success:
        print("\n🎉 Conversion completed successfully!")
        print("You can now use the TensorFlow.js model in your web application.")
    else:
        print("\n❌ Conversion failed. Trying direct method...")