import os
import cv2

image_dir = 'Dataset/DIV8K/test'
image_names = os.listdir(image_dir)

for image_name in image_names:
    path = os.path.join(image_dir, image_name)
    image = cv2.imread(path)

    