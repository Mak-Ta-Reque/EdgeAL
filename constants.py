import os
DATASET = "duke"
HDD_DATASET_ROOT = "/mnt/sda/abka03-data/datasets/medical/segmentation/edgeal/datasets"
SSD_DATASET_ROOT = "/mnt/sda/abka03-data/datasets/medical/segmentation/edgeal/datasets"
RUNS = "/mnt/sda/abka03-data/checkpoints/submission"
DEPTH_WIDTH = 224
DEPTH_HEIGHT = 224
MC_STEPS = 20
MC_DROPOUT_RATE = 0.25
EDGE_THRESHOLD = 1.0
COVERAGE_IGNORE_THRESHOLD = 0.35
VISUALIZATION = True
SEEDS_SUPERPIXEL = "seeds-revised/bin/reseeds_cli"
N_CLASSES = 9
IN_CHANNELS = 1
INITAI_SEED_DATA = 0.02
# UMN dataset has 2 classes
# Duke has 9
# Aroi has 8
