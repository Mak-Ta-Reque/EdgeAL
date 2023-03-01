import sys
import os
current_dir = file_dir = "/".join(os.path.dirname(__file__).split("/")[:-2])
sys.path.append(current_dir)
import constants
from tqdm import tqdm
from pathlib import Path
import random
from multiprocessing import Pool
import multiprocessing as mp
from skimage.io import imread
import glob
from PIL import Image
raw_files_location = os.path.join(constants.HDD_DATASET_ROOT, constants.DATASET, "raw/selections")
print(raw_files_location)
#"/mnt/sda/haal02-data/active_learning/datasets/idrid-dataset/raw/selections"
seeds_segmentor_path = "seeds-revised/bin/reseeds_cli"
splits_path = Path(os.path.join(constants.HDD_DATASET_ROOT, constants.DATASET, "selections"))#Path('/mnt/sda/haal02-data/active_learning/datasets/idrid-dataset')

def read_scene_list(path):
    with open(path, "r") as fptr:
        return [x.strip() for x in fptr.readlines() if x.strip() != ""]

def write_frames_list(name, paths):
    with open((splits_path / f"{name}_frames.txt"), "w") as fptr:
        for x in paths:
            fptr.write(f"{x}\n")

def create_splits():
    print(splits_path)
    train_scenes = read_scene_list(str(splits_path / "scenes_train.txt"))
    val_scenes = read_scene_list(str(splits_path / "scenes_val.txt"))
    test_scenes = read_scene_list(str(splits_path / "scenes_test.txt"))

    train_frames = []
    val_frames = []
    test_frames = []

    for x in (raw_files_location / "color").iterdir():
        # For scannet names - format = {sceneid_rescanid}_frameid
        if "_".join(x.name.split("_")[0:2]) in train_scenes:
            train_frames.append(x.name.split(".")[0])
        elif "_".join(x.name.split("_")[0:2]) in val_scenes:
            val_frames.append(x.name.split(".")[0])
        elif "_".join(x.name.split("_")[0:2]) in test_scenes:
            test_frames.append(x.name.split(".")[0])

    print(len(train_frames), len(val_frames), len(test_frames))

    write_frames_list("train", train_frames)
    write_frames_list("val", val_frames)
    write_frames_list("test", test_frames)

def call(*popenargs, **kwargs):
    from subprocess import Popen, PIPE
    kwargs['stdout'] = PIPE
    kwargs['stderr'] = PIPE
    p = Popen(popenargs, **kwargs)
    stdout, stderr = p.communicate()
    if stdout:
        for line in stdout.decode("utf-8").strip().split("\n"):
            print(line)
    if stderr:
        for line in stderr.decode("utf-8").strip().split("\n"):
            print(line)
    return p.returncode

def create_superpixel_segmentations():
    all_args = "--spatial-weight 0.2 --superpixels 40 --iterations 10 --confidence 0.001"
    colordir_src = os.path.join(raw_files_location, "color")
    args = all_args.split(" ")
    
    with Pool(processes=8) as pool:
        arg_list = []
        for c in tqdm(os.listdir(colordir_src)):
            color_base_name = c.split(".")[0]
            spx_target = os.path.join(raw_files_location, "superpixel", f"{color_base_name}.png")
            print(spx_target)
            arg_list.append((seeds_segmentor_path, "--input", os.path.join(colordir_src, c), "--output", spx_target,
                             args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], "--index"))
        pool.starmap(call, tqdm(arg_list))
    


def create_split_mixed():
    list_of_frames = [x.split(".")[0] for x in os.listdir(os.path.join(raw_files_location, "color"))]
    train_frames = [list_of_frames[i] for i in random.sample(range(len(list_of_frames)), int(0.60 * len(list_of_frames)))]
    remaining_frames = [x for x in list_of_frames if x not in train_frames]
    val_frames = [remaining_frames[i] for i in random.sample(range(len(remaining_frames)), int(0.15 * len(list_of_frames)))]
    test_frames = [x for x in remaining_frames if x not in val_frames]
    print(len(train_frames)/len(list_of_frames), len(val_frames)/len(list_of_frames), len(test_frames)/len(list_of_frames))
    with open(splits_path / "train_frames.txt", "w") as fptr:
        for x in train_frames:
            fptr.write(x+"\n")
    with open(splits_path / "val_frames.txt", "w") as fptr:
        for x in val_frames:
            fptr.write(x+"\n")
    with open(splits_path / "test_frames.txt", "w") as fptr:
        for x in test_frames:
            fptr.write(x+"\n")

def create_seed_set():
    train_frames = (splits_path / "train_frames.txt").read_text().split()
    seed_frames = [train_frames[i] for i in random.sample(range(len(train_frames)), int(constants.INITAI_SEED_DATA * len(train_frames)))]
    print(len(seed_frames))
    print(splits_path)
    with open(splits_path / "seedset_0_frames.txt", "w") as fptr:
        for x in seed_frames:
            fptr.write(x.split(".")[0]+"\n")

def resize_image(path):
    image = Image.open(path[0])
    # Resize the image
    resized_image = image.resize(path[1])
    # Save the resized image to a file
    resized_image.save(path[0])

def resize_depth(size):
    superpixel_path = os.path.join(raw_files_location, "superpixel")
    png_files = glob.glob(os.path.join(superpixel_path, '*.png'))
    argument = list(zip(png_files, len(png_files)*[size]))
    #sprint(list(argument))

    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    
    # Perform the computation in parallel
    results = pool.map(resize_image,  argument)
    
def resize_superpixel(size):
    superpixel_path = os.path.join(raw_files_location, "depth")
    png_files = glob.glob(os.path.join(superpixel_path, '*.png'))
    argument = list(zip(png_files, len(png_files)*[size]))
    #sprint(list(argument))

    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    
    # Perform the computation in parallel
    results = pool.map(resize_image,  argument)

def resize_pose(size):
    superpixel_path = os.path.join(raw_files_location, "pose")
    png_files = glob.glob(os.path.join(superpixel_path, '*.png'))
    argument = list(zip(png_files, len(png_files)*[size]))
    #sprint(list(argument))

    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    
    # Perform the computation in parallel
    results = pool.map(resize_image,  argument)
    


if __name__=='__main__':
    create_superpixel_segmentations()
    resize_superpixel((constants.DEPTH_WIDTH, constants.DEPTH_HEIGHT))
    resize_depth((constants.DEPTH_WIDTH, constants.DEPTH_HEIGHT))
    #resize_pose((constants.DEPTH_WIDTH, constants.DEPTH_HEIGHT))
    
    #create_split_mixed() # or create_split()
    create_seed_set()
    
