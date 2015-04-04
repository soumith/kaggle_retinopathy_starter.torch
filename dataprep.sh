#!/bin/bash
cd data
#7z e train.zip.001
#mkdir -p train
#mv *.jpeg train/
#7z e test.zip.001
#mkdir -p test
#mv *.jpeg test/

find . -name "*.jpeg" | xargs -I {} echo 'convert {} -resize "256^>" {} -quality 100 '>commands.txt
cat commands.txt | parallel

unzip trainLabels.csv.zip
th split.lua crossvalidation_splits/val_1.txt
