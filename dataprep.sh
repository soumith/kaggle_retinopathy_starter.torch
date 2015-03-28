#!/bin/bash
cd data
#7z e train.zip.001
#mkdir -p train
#mv *.jpeg train/
#find . -name "*.jpeg" | xargs -I {} convert {} -resize "256^>" {}
unzip trainLabels.csv.zip

th split.lua crossvalidation_splits/val_1.txt
