import sys
import os
current_dir = file_dir = "/".join(os.path.dirname(__file__).split("/")[:-2])
sys.path.append(current_dir)

import argparse

import glob
import numpy as np
from PIL import Image
import constants
import selections

# Create the argument parser
parser = argparse.ArgumentParser(description='Example program that reads a file path.')

# Add the argument
parser.add_argument('filepath', type=str, help='the file path to read')

# Parse the arguments
args = parser.parse_args()

def create_path(path):
    os.makedirs(path, exist_ok=True)
    print("Subdirectory created:", path)

def save_image_to_dir(list_npy, path):
    for item in list_npy:
        np_file = np.uint8(np.load(item))
        pil_image = Image.fromarray(np_file)
        #pil_image = pil_image.resize((constants.DEPTH_WIDTH, constants.DEPTH_HEIGHT))
        pil_image.save(os.path.join(path, item.split("/")[-1].split(".")[0]+".png"))

def save_mask_to_dir(list_npy, path):
    for item in list_npy:
        np_file = np.uint8(np.load(item))
        pil_image = Image.fromarray(np_file, dtype=np.uint8)
        #pil_image = pil_image.resize((constants.DEPTH_WIDTH, constants.DEPTH_HEIGHT))
        pil_image.save(os.path.join(path, item.split("/")[-1].split(".")[0]+".png"))




# Create selection directory
selection_path = os.path.join(args.filepath, "selections")
create_path(selection_path)
# Set the directory path to create
raw_selection_path = os.path.join(args.filepath, "raw/selections")
create_path(raw_selection_path)

folders = ["color", "label", "pose", "depth", "superpixel", "coverage_superpixel"]
for f in folders:
    path = os.path.join(raw_selection_path, f)
    create_path(path=path)

label = os.path.join(raw_selection_path, "label")
color = os.path.join(raw_selection_path, "color")

splits = ["train", "test", "val"]
for sp in splits:
    im_path = os.path.join(args.filepath, sp, "images")
    msk_path = os.path.join(args.filepath, sp, "masks")
    list_image = glob.glob(im_path+ '/*.npy')
    list_mask = glob.glob(msk_path+ '/*.npy')
    save_image_to_dir(list_image, color)
    save_image_to_dir(list_mask, label)

    list_names = [os.path.split(item)[-1].split(".")[0] for item in list_image]

    file_path = os.path.join(selection_path, f"{sp}_frames.txt")
# Open the file in write mode
    with open(file_path, 'w') as file:
        # Iterate over the list and write each item to a new line
        for item in list_names:
            file.write(item + '\n')
        print("List written to file:", file_path)