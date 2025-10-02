from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import pytesseract
import regex as re
import numpy as np
import requests
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv, dotenv_values
from pathlib import Path

# Load environment variables from .env file
# Get the parent directory (where .env file is located)
env_path = Path(__file__).parent.parent / '.env.new'
print(f"Debug: .env path = {env_path}")
print(f"Debug: .env exists = {env_path.exists()}")

# Try loading with explicit path and verbose output
from dotenv import dotenv_values
env_vars = dotenv_values(env_path)
print(f"Debug: dotenv_values result = {env_vars}")

# Load the variables
load_result = load_dotenv(dotenv_path=env_path, verbose=True, override=True)
print(f"Debug: load_dotenv result = {load_result}")

# Also try loading all environment variables to see what's available
print(f"Debug: All env vars starting with CLOUDINARY:")
for key, value in os.environ.items():
    if key.startswith('CLOUDINARY'):
        print(f"  {key} = {value}")

# --- Cloudinary Configuration ---
# IMPORTANT: These will be set as Environment Variables in Vercel, not hardcoded.
print(f"Debug: CLOUDINARY_CLOUD_NAME = {os.getenv('CLOUDINARY_CLOUD_NAME')}")
print(f"Debug: CLOUDINARY_API_KEY = {os.getenv('CLOUDINARY_API_KEY')}")
print(f"Debug: CLOUDINARY_API_SECRET = {'***' if os.getenv('CLOUDINARY_API_SECRET') else None}")

# Fallback: use the values from dotenv_values if os.getenv fails
cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME') or env_vars.get('CLOUDINARY_CLOUD_NAME')
api_key = os.getenv('CLOUDINARY_API_KEY') or env_vars.get('CLOUDINARY_API_KEY')
api_secret = os.getenv('CLOUDINARY_API_SECRET') or env_vars.get('CLOUDINARY_API_SECRET')

print(f"Debug: Final values - cloud_name={cloud_name}, api_key={api_key}, api_secret={'***' if api_secret else None}")

cloudinary.config(
  cloud_name = cloud_name,
  api_key = api_key,
  api_secret = api_secret,
  secure = True
)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Pydantic Models for Request and Response ---
class BlurRequest(BaseModel):
    image_url: str
    blur_options: list = ['email', 'phone', 'name', 'address', 'pan', 'aadhaar']

class BlurResponse(BaseModel):
    blurred_image_url: str

# --- Core Image Processing Logic ---
# This is your function, modified to accept a NumPy array (image data) instead of a file path.
def process_image_data(image, blur_options: list):
    try:
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(status_code=500, detail="Tesseract is not installed or not in PATH on the server.")

    patterns = {
        'email': {'regex': r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}', 'type': 'direct'},
        'phone': {'regex': r'(?:\+91[\s-]?)?\b[6-9]\d{9}\b', 'type': 'direct'},
        'pan': {'keywords': ['pan'], 'regex': r'\b[a-z]{5}[0-9]{4}[a-z]{1}\b', 'type': 'keyword'},
        'aadhaar': {'keywords': ['aadhaar', 'uid'], 'regex': r'\b\d{4}\s?\d{4}\s?\d{4}\b', 'type': 'keyword'},
        'name': {'keywords': ['name', 'mr', 'mrs', 'ms'], 'regex': r'\b[a-z]+\b', 'type': 'keyword'},
        'address': {'keywords': ['address'], 'type': 'address'}
    }

    boxes_to_blur = []
    num_boxes = len(ocr_data['text'])

    for i in range(num_boxes):
        text = ocr_data['text'][i].strip().lower()
        if not text or int(ocr_data['conf'][i]) < 40:
            continue

        for option in blur_options:
            p_info = patterns.get(option, {})
            if p_info.get('type') == 'direct' and re.fullmatch(p_info['regex'], text, re.IGNORECASE):
                boxes_to_blur.append(i)
                break
            elif p_info.get('type') == 'keyword' and text in p_info['keywords']:
                current_line = ocr_data['line_num'][i]
                data_index = -1
                for j in range(i + 1, num_boxes):
                    if ocr_data['line_num'][j] != current_line: break
                    next_text = ocr_data['text'][j].strip()
                    if next_text and next_text not in [':', '-']:
                        data_index = j
                        break
                if data_index != -1 and re.fullmatch(p_info['regex'], ocr_data['text'][data_index], re.IGNORECASE):
                    boxes_to_blur.append(data_index)
                break
            elif p_info.get('type') == 'address' and text in p_info['keywords']:
                current_line = ocr_data['line_num'][i]
                for j in range(i + 1, num_boxes):
                    if ocr_data['line_num'][j] == current_line:
                        if ocr_data['text'][j].strip(): boxes_to_blur.append(j)
                    elif ocr_data['line_num'][j] > current_line: break
                next_line_num = current_line + 1
                for j in range(i + 1, num_boxes):
                    if ocr_data['line_num'][j] == next_line_num:
                        if ocr_data['text'][j].strip(): boxes_to_blur.append(j)
                    elif ocr_data['line_num'][j] > next_line_num: break
                break

    for index in set(boxes_to_blur):
        (x, y, w, h) = (ocr_data['left'][index], ocr_data['top'][index], ocr_data['width'][index], ocr_data['height'][index])
        if w > 0 and h > 0:
            roi = image[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (23, 23), 30)
            image[y:y+h, x:x+w] = blurred_roi

    return image

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Blurring API"}

@app.post("/blur", response_model=BlurResponse)
async def blur_image(request: BlurRequest):
    try:
        # 1. Download the image from the provided Cloudinary URL
        response = requests.get(request.image_url)
        response.raise_for_status() # Raise an exception for bad status codes
        image_data = np.frombuffer(response.content, np.uint8)
        
        # 2. Decode the image data into an OpenCV-readable format
        original_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if original_image is None:
            raise HTTPException(status_code=400, detail="Could not decode image from the provided URL.")

        # 3. Process the image using your core logic
        blurred_image = process_image_data(original_image, request.blur_options)

        # 4. Encode the blurred image back to a byte stream for uploading
        _, buffer = cv2.imencode('.png', blurred_image)
        image_bytes = buffer.tobytes()

        # 5. Upload the blurred image to Cloudinary
        upload_result = cloudinary.uploader.upload(
            image_bytes,
            folder="blurred_images" # Optional: specify a folder in Cloudinary
        )
        
        # 6. Return the secure URL of the newly uploaded image
        return {"blurred_image_url": upload_result['secure_url']}

    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="Invalid image URL or could not fetch image.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")