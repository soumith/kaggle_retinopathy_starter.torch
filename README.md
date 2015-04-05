# kaggle_retinopathy_starter.torch
A starter kit in Torch for Kaggle Diabetic Retinopathy Detection.
It showcases:
- Classification
- Regression
- Metric Learning (Siamese and triplet networks)
- Averaging model ensembles

## What else?
- 1-GPU or multi-GPU convolution neural networks
- multi-threaded data loading (data is loaded in compressed-form into memory and decompressed + jittered on the fly)
- test script to take your trained model (or models) and produce Kaggle-compatible CSV file ready for upload

## Getting started
- Install torch + dependencies, instructions are here: [INSTALL.md](INSTALL.md)
- If you already have torch installed, update it and it's packages. These scripts use some recently added features.

- Run script with:
```
th main.lua
```

##Ideas to try
- Maxout Networks: http://arxiv.org/abs/1302.4389
- All Convolution Net: http://arxiv.org/abs/1412.6806
- Separable Convolution filters
- Dropout/DropConnect
- PReLU: https://github.com/torch/nn/blob/master/doc/transfer.md#prelu
- Batch Normalization: https://github.com/torch/nn/blob/master/doc/simple.md#batchnormalization
- L2 Pooling: https://github.com/torch/nn/blob/master/SpatialLPPooling.lua
- Different data augmentation
- Polar/Log-polar images: https://github.com/torch/image#res-imagepolardst-src-interpolation-mode
- Different color spaces: https://github.com/torch/image#color-space-conversions
- All of Sander Deileman's tricks:
  - http://benanne.github.io/2014/04/05/galaxy-zoo.html
  - http://benanne.github.io/2015/03/17/plankton.html

##Broad Strategy
- Train several models
- Average the network predictions over many of these models
- For each image, average the prediction over several crops and orientations of the image (for example at test time, every image's prediction is the average of 4 rotations of the image)
- Win
