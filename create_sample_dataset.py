"""
create_sample_dataset.py
------------------------
Generates a small synthetic dataset for testing the transfer learning pipeline.
Creates two classes: 'class_a' (reddish circles) and 'class_b' (bluish rectangles).
"""

import os
from PIL import Image, ImageDraw
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "custom_dataset")

CLASSES = ["class_a", "class_b"]
IMAGES_PER_CLASS = 20
IMG_SIZE = (224, 224)

def generate_sample_dataset():
    np.random.seed(42)
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        print(f"Generating {IMAGES_PER_CLASS} sample images in {cls_dir}...")
        
        for i in range(IMAGES_PER_CLASS):
            img = Image.new("RGB", IMG_SIZE, color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            if cls == "class_a":
                # Draw reddish shapes
                r = np.random.randint(180, 255)
                g = np.random.randint(20, 80)
                b = np.random.randint(20, 80)
                x0 = np.random.randint(20, 80)
                y0 = np.random.randint(20, 80)
                x1 = np.random.randint(140, 200)
                y1 = np.random.randint(140, 200)
                draw.ellipse([x0, y0, x1, y1], fill=(r, g, b))
            else:
                # Draw bluish shapes
                r = np.random.randint(20, 80)
                g = np.random.randint(20, 80)
                b = np.random.randint(180, 255)
                x0 = np.random.randint(30, 90)
                y0 = np.random.randint(30, 90)
                x1 = np.random.randint(130, 190)
                y1 = np.random.randint(130, 190)
                draw.rectangle([x0, y0, x1, y1], fill=(r, g, b))
                
            img_path = os.path.join(cls_dir, f"sample_{i+1:03d}.jpg")
            img.save(img_path, "JPEG")
            
    print("[SUCCESS] Sample dataset generated successfully.")

if __name__ == "__main__":
    generate_sample_dataset()
