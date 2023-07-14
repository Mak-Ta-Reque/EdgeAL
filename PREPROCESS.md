## Preprocess data
The data should follow the directroy structre from ```README.md```

For EdgeAL algorithm you need ```edge```, ```dct```, ```info``` ```data```

Before traing you need to create ```superpixel``` directory and ```extended_superpixel``` directory

You can generate superpixel using [this SEEDS implementation](https://github.com/Mak-Ta-Reque/seeds-revised) 
# How?
Follow the instuction of SEEDS implementation and add the binary directory```seeds-revised/bin/reseeds_cli``` to constants.py

# Running preprcessing script
`python EdgeAL_submission/dataset/preprocessing-scripts/selections.py` It generates train test val segments also generate superpixels files
`python EdgeAL_submission/dataset/preprocessing-scripts/to_lmdb.py` It creates a compact data file from the folder
`python EdgeAL_submission/utils/sovel_dct.py` It creates sobel segmentations and dct of image and save it in edge and poses files. Alternative of depeth and pose information
`python EdgeAL_submission/utils/superpixel_projections.py` It geneates the superpixel overlap files



For training follow read me file.


# Adding new dataset
name the datset `dataset-name` and follow the structure form the README.md
Add the path and name (both are unique) in constants
Add Width and Heght of images 
and set additional information 
Add the dataset name in the argument_parser

And finally train with the instruction given in README.md



