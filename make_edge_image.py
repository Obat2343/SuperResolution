from PIL import Image
import numpy as np
import os
import sys
import cv2
from tqdm import tqdm

def edge_image(root_dir,new_filedir):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    for Cur_dir, _, filelist in os.walk(root_dir,followlinks=True):
        for filename in tqdm(filelist):
            filepath = os.path.join(Cur_dir,filename)
            new_file_path = os.path.join(new_filedir,filename)
            head, ext = os.path.splitext(filename)
            if ext in ['.png','.jpg','.JPG','.jpeg','.webp']:
                img = cv2.imread(filepath)
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                img = cv2.medianBlur(img,5)
                img = cv2.medianBlur(img,3)
                img = cv2.Laplacian(img, cv2.CV_64F, ksize=3, scale=5)
                img = cv2.convertScaleAbs(img)
                img = cv2.medianBlur(img,3)
                # new_file_path = os.path.join(new_filedir,"{}_{}{}".format(head,i,ext))
                cv2.imwrite(new_file_path, img)

if __name__ == '__main__':
    edge_image(sys.argv[1],sys.argv[2])