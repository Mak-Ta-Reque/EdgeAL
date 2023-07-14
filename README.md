# EdgeAL: Active Learning with Edge information

This repository contains the implementation for the paper:
Kadir M.A, Alam H. T, Sonntag D., ["EdgeAL: An Edge Estimation Based Active
Learning Approach for OCT Segmentation"]() [MICAAI2023]


## Running

#### Arguments

```
train_active.py [-h]    [--model {deeplab, unet, y_net_gen, y_net_gen_ffc}]
                        [--backbone {resnet, xception, drn, mobilenet}]
                       [--out-stride OUT_STRIDE]
                       [--dataset {duke, UMN, AROI}]
                       [--workers N] [--base-size BASE_SIZE]
                       [--sync-bn SYNC_BN] [--loss-type {'ce', 'focal', 'ce+dice'}]
                       [--epochs N] [--batch-size N] [--use-balanced-weights]
                       [--lr LR] [--lr-scheduler {step}]
                       [--optimizer {SGD,Adam}] [--step-size STEP_SIZE]
                       [--use-lr-scheduler] [--momentum M] [--weight-decay M]
                       [--nesterov] [--gpu-ids GPU_IDS] [--seed S]
                       [--checkname CHECKNAME] [--eval-interval EVAL_INTERVAL]
                       [--memory-hog]
                       [--max-iterations MAX_ITERATIONS]
                       [--active-selection-size ACTIVE_SELECTION_SIZE]
                       [--region-size REGION_SIZE]
                       [--region-selection-mode REGION_SELECTION_MODE]
                       [--view-entropy-mode {soft,vote,mc_dropout}]
                       [--active-selection-mode {random, 'edgeal_region'}]
                       [--superpixel-dir SUPERPIXEL_DIR]
                       [--superpixel-coverage-dir SUPERPIXEL_COVERAGE_DIR]
                       [--superpixel-overlap SUPERPIXEL_OVERLAP]
                       [--start-entropy-threshold START_ENTROPY_THRESHOLD]
                       [--entropy-change-per-selection ENTROPY_CHANGE_PER_SELECTION]
```

Run `--help` for more details.


#### Example commands


##### EdgeAL
```sh
# sample dataset
python train_active.py --model y_net_gen_ffc --g-ratio 0.5 --dataset duke --workers 2 --epochs 5 --eval-interval 2 --batch-size=10 --lr 5e-4 --weight-decay 1e-4 --optimizer Adam --use-lr-scheduler --lr-scheduler step --step-size 100 --checkname view_entropy_ynet+fcc_0_lr-0.0004_bs-6_ep-60_wb-0_lrs-0_240x240ce+dice --base-size 224,224 --loss-type ce+dice --max-iterations 10 --active-selection-size 7 --active-selection-mode edgeal_region --region-selection-mode superpixel

##### Random
```sh
python train_active.py --model y_net_gen_ffc --g-ratio 0.5 --dataset duke --workers 2 --epochs 100 --eval-interval 1 --batch-size=10 --lr 5e-4 --weight-decay 1e-4 --optimizer Adam --use-lr-scheduler --lr-scheduler step --step-size 100 --checkname random_ynet+fcc_0_lr-0.0004_bs-6_ep-60_wb-0_lrs-0_240x240focalsoft --base-size 224,224 --loss-type ce+dice --max-iterations 10 --active-selection-size 7 --active-selection-mode random

```


## Files

Overall code structure is as follows: 

| File / Folder | Description |
| ------------- |-------------| 
| train_active.py | Training script for active learning methods | 
| train.py | Training script for full dataset training | 
| constants.py | Constants used across the code |
| argument_parser.py | Arguments parsing code |
| active_selection | Implementation of our method and other active learning methods for semantic segmentation |
| dataloader | Dataset classes |
| dataset | Train/test splits and raw files of datasets |
| model | DeeplabV3+ implementation (inspired from [here](https://github.com/jfzhang95/pytorch-deeplab-xception))|
| utils| Misc utils |

The datasets must follow the following structure

```
dataset # root dataset directory
├── dataset-name
    ├── raw
        ├── selections
            ├── color # rgb frames
            ├── label # ground truth maps
            ├── depth # depth maps
            ├── pose # camera extrinsics for each frame
            ├── info # camera intrinsics
            ├── superpixel # superpixel maps
            ├── coverage_superpixel # coverage maps
    ├── selections
        ├── seedset_0_frames.txt # seed set
        ├── train_frames.txt 
        ├── val_frames.txt
        ├── test_frames.txt
    ├── dataset.lmdb # rgb frames + labels in lmdb format
```

A small example dataset is provided with this repository in [`dataset/duke-sample`](https://github.com/nihalsid/ViewAL/tree/master/dataset/duke-sample).

## Data Generation

To use this repository datasets must be in the structure described in last section. For creating the lmdb database, seed set, train / test splits and superpixel maps check helper scripts in [`dataset/preprocessing-scripts`](https://github.com/nihalsid/ViewAL/tree/master/dataset/preprocessing-scripts). We use [this SEEDS implementation](https://github.com/Mak-Ta-Reque/seeds-revised) for generating superpixels (check [this](https://github.com/nihalsid/ViewAL/issues/4) issue for troubleshooting). Further, to generate superpixel coverage maps (`coverage_superpixel`) check [`utils/superpixel_projections.py`](https://github.com/nihalsid/ViewAL/blob/master/utils/superpixel_projections.py). 

## Citation


## Similar research
Cost-Effective Active Learning for Melanoma Segmentation
```https://github.com/imatge-upc/medical-2017-nipsw```
```https://github.com/marc-gorriz/CEAL-Medical-Image-Segmentation```

## Dataset sources
``` https://grand-challenge.org```
