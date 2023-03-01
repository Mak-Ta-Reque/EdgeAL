import os
DATASET = "UMN"
HDD_DATASET_ROOT = "/mnt/sda/haal02-data/active_learning/datasets/"
SSD_DATASET_ROOT = "/mnt/sda/haal02-data/active_learning/datasets/"
RUNS = "/mnt/sda/abka03-data/checkpoints/umn_small_increase"
DEPTH_WIDTH = 224
DEPTH_HEIGHT = 224
MC_STEPS = 20
MC_DROPOUT_RATE = 0.25
WORLD_DISTANCE_THRESHOLD = 1.0
COVERAGE_IGNORE_THRESHOLD = 0.35
VISUALIZATION = True
SEEDS_SUPERPIXEL = "seeds-revised/bin/reseeds_cli"
N_CLASSES = 2
IN_CHANNELS = 1
INITAI_SEED_DATA = 0.02
# UMN dataset has 2 classes
# Duke has 9
# Aroi has 8