# kaggle_retinopathy_starter.torch
A starter kit in Torch for Kaggle Diabetic Retinopathy Detection.

## What?
- CPU, 1-GPU or multi-GPU convolution neural networks
- multi-threaded data loading (data is loaded in compressed-form into memory and decompressed + jittered on the fly)
- test script to take your trained model and produce Kaggle-compatible CSV file ready for upload

## Getting started
- Install torch + dependencies, instructions are here: [INSTALL.md](INSTALL.md)

- Run script with:
```
th main.lua
```

