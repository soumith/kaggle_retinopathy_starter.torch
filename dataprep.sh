#!/bin/bash
cd data

## UNZIP Files
7z e train.zip.001
mkdir -p train
mv *.jpeg train/
7z e test.zip.001
mkdir -p test
mv *.jpeg test/

## Resize down to 256
find . -name "*.jpeg" | xargs -I {} echo 'convert {} -resize "256^>" -quality 100 {}'>commands.txt
cat commands.txt | parallel

mv train train_256
mv test test_256

# resize down to 64 as well
cp -r train_256 train_64
cp -r test_256 test_64
find train_256 -name "*.jpeg" | xargs -I {} echo 'convert {} -resize "256^>" -quality 100 {}'>commands.txt
cat commands.txt | parallel
find test_256 -name "*.jpeg" | xargs -I {} echo 'convert {} -resize "256^>" -quality 100 {}'>commands.txt
cat commands.txt | parallel

unzip trainLabels.csv.zip
th split.lua crossvalidation_splits/val_1.txt
