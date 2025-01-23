import base64
import os
from flask import Flask, request, jsonify
from gradio_client import Client, file
from werkzeug.utils import secure_filename
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Add this configuration
if os.environ.get('RENDER'):
    # Configure for Render.com
    UPLOAD_FOLDER = "/tmp/uploads"  # Use /tmp for Render
else:
    UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = Client("Nymbo/Virtual-Try-On")

def image_to_base64(image_path):
    """Convert image to base64 encoded string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def preprocess_image(image_path, size=(512, 512)):
    """Resize image to the given size"""
    img = Image.open(image_path)
    img = img.resize(size)
    img.save(image_path)

def correct_aspect_ratio(image_path, target_size=(512, 512)):
    """Ensure output images have the correct aspect ratio"""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img.save(image_path)

@app.route('/tryon', methods=['POST'])
def try_on():
    try:
        # Validate uploaded files
        if 'background_image' not in request.files or 'garment_image' not in request.files:
            return jsonify({"error": "Missing image files"}), 400

        background_file = request.files['background_image']
        garment_file = request.files['garment_image']
        garment_description = request.form.get('garment_description', 'A stylish garment')

        # Save files temporarily
        background_path = os.path.join(UPLOAD_FOLDER, secure_filename(background_file.filename))
        garment_path = os.path.join(UPLOAD_FOLDER, secure_filename(garment_file.filename))
        background_file.save(background_path)
        garment_file.save(garment_path)

        # Preprocess images
        preprocess_image(background_path)
        preprocess_image(garment_path)

        # Call the Gradio Client API
        result = client.predict(
            {
                "background": file(background_path),
                "layers": [],
                "composite": None
            },
            garm_img=file(garment_path),
            garment_des=garment_description,
            is_checked=True,  # Ensure alignment
            is_checked_crop=True,  # Enable cropping for better results
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        # Extract output paths
        output_image_path, masked_image_path = result

        # Correct output aspect ratio
        correct_aspect_ratio(output_image_path)
        correct_aspect_ratio(masked_image_path)

        # Convert images to base64
        output_image_base64 = image_to_base64(output_image_path)
        masked_image_base64 = image_to_base64(masked_image_path)

        # Clean up uploaded files
        os.remove(background_path)
        os.remove(garment_path)

        return jsonify({
            "output_image": output_image_base64,
            "masked_image": masked_image_base64
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
