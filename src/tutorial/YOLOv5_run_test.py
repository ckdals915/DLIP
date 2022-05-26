import torch
import cv2
import random
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
if torch.cuda.is_available(): print(f'Device name: {torch.cuda.get_device_name(0)}') 

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Image preparation from URL . images link
img_URL = [           
           "https://user-images.githubusercontent.com/23421059/168719874-be48ef28-954c-4a4c-a048-1e11699e0b56.png",
           ]

imgs = []
img_L = len(img_URL)

# Append these 3 images and save 
for i in range(img_L):
  imgName = f"{i}.jpg"
  torch.hub.download_url_to_file(img_URL[i],imgName)  # download 2 images
  # imgs.append(Image.open(fileName))  # PIL image
  imgs.append(cv2.imread(imgName)[:,:,::-1]) # OpenCV image (BGR to RGB)

# Run Inference
results = model(imgs)

# Print Results
results.print()

# Save Result images with bounding box drawn
results.save()  # or .show()

# Select a random test image
randNo = random.choice(range(img_L))
print(f"Selected Image No = {randNo}\n\n")

# Print the Bounding Box result:  6 columns
# Column (1~4) Coordinates of TL, BR corners (5) Confidence (6) Class ID
print(results.xyxy[randNo],'\n')  # imgs predictions (tensor)

# Print the Bounding Box result using Pandas
print(results.pandas().xyxy[randNo],'\n')  # imgs predictions (pandas)

# Show result image
cv2.imshow("result", (results.imgs[randNo])[:,:,::-1])
cv2.waitKey(0)