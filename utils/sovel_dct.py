import sys
import os
current_dir = file_dir = "/".join(os.path.dirname(__file__).split("/")[:-1])
sys.path.append(current_dir)
#import constants
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
import cv2
import constants
color = os.path.join(constants.HDD_DATASET_ROOT, constants.DATASET , "raw/selections/color") #os.path.join(constants.HDD_DATASET_ROOT, "AROI/raw/selections/color")
def sobel(path):
    new_path  = os.path.join(constants.HDD_DATASET_ROOT, constants.DATASET, "raw/selections/depth", path)
    path = os.path.join(color, path)
    # Load the image
    img = Image.open(path).convert("L")
    img = np.array(img)
    # Apply the Sobel operator to the image
    sobel_x = ndimage.sobel(img, axis=0)
    sobel_y = ndimage.sobel(img, axis=1)
    sobel_m = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
    sobel_m = np.uint8(255 * sobel_m / np.max(sobel_m))
    #print(sobel_m.shape)
    im = Image.fromarray(sobel_m)
    im.save(new_path)

def dct(path):
    new_path  = os.path.join(constants.HDD_DATASET_ROOT, constants.DATASET, "raw/selections/pose", path)
    path = os.path.join(color, path)
    img = Image.open(path).convert("L")
    img = np.array(img)
    dct = cv2.dct(np.float32(img))
    dct = np.uint8(255 * dct / np.max(dct))
    #new_path = os.path.join("/".join(path[0:-2]), "pose", path[-1])
    #print(new_path)
    im = Image.fromarray(dct)
    im.save(new_path)


def main():
    files = os.listdir(color)

    # Filter the list to keep only the .jpg files
    jpg_files = [file for file in files if file.endswith(".png")]
    print("Total files: ", len(jpg_files))
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)

# Perform the computation in parallel
    results = pool.map(sobel, jpg_files)
    print("Sobel finished")
    results = pool.map(dct, jpg_files)
    print("Dct finishd")



if __name__ == '__main__':
    main()
