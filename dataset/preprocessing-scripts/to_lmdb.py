import sys
import os
current_dir = file_dir = "/".join(os.path.dirname(__file__).split("/")[:-2])
sys.path.append(current_dir)
import pickle
import lmdb
import glob
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import constants


def to_lmdb(root_path, lmdb_path, image_size):

    image_paths = [x.split(".")[0] for x in os.listdir(os.path.join(root_path, "color"))]
    print('#images: ', len(image_paths))
    print("Generate LMDB to %s" % lmdb_path)

    #image_size = Image.open(os.path.join(root_path, "color", f'{image_paths[0]}.png')).size
    pixels = image_size[0] * image_size[1] * len(image_paths)

    print("Pixels in split: ", pixels)

    map_size = pixels * 4 + 1500 * pixels * 4 
    #map_size = 50 *  pixels * 4 #pixels * 4 + 1500 * 320 * 240 * 4

    print("Estimated Size: ", map_size / (320 * 240* len(image_paths)))

    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir, map_size=map_size, readonly=False, meminit=False, map_async=True, writemap=True)

    txn = db.begin(write=True)

    key_list = []

    for idx, path in tqdm(enumerate(image_paths)):
        jpg_path = os.path.join(root_path, 'color', f'{path}.png')
        png_path = os.path.join(root_path, 'label', f'{path}.png')
        if constants.IN_CHANNELS == 3:
            image = Image.open(jpg_path).convert('RGB')
        else:
            image = Image.open(jpg_path)
        #image = image.resize(image_size,Image.ANTIALIAS)
        image = np.array(image, dtype=np.uint8)
        label = Image.open(png_path)
        #label = label.resize(image_size,Image.ANTIALIAS)

        label = np.array(label, dtype=np.uint8)
        #label -= 1
        #label[label == -1] = 255
        #label[label > constants.N_CLASSES] = 255

        txn.put(u'{}'.format(path).encode('ascii'), pickle.dumps(np.dstack((image, label)), protocol=3))
        key_list.append(path)

    print('Committing..')
    txn.commit()

    print('Writing keys..')
    keys = [u'{}'.format(k).encode('ascii') for k in key_list]

    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys, protocol=3))
        txn.put(b'__len__', pickle.dumps(len(keys), protocol=3))

    print('Syncing..')
    db.sync()
    db.close()


if __name__ == '__main__':
    image_size = (constants.DEPTH_WIDTH, constants.DEPTH_HEIGHT ) #  320,160
    PATH_TO_SELECTIONS = os.path.join(constants.HDD_DATASET_ROOT, constants.DATASET, "raw/selections") 
    #"/mnt/sda/haal02-data/active_learning/datasets/idrid-dataset/raw/selections"
    PATH_TO_LMDB = os.path.join(constants.HDD_DATASET_ROOT, constants.DATASET, "dataset.lmdb") #"/mnt/sda/haal02-data/active_learning/datasets/idrid-dataset/dataset.lmdb"
    to_lmdb(PATH_TO_SELECTIONS, PATH_TO_LMDB, image_size)
