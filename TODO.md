- Classification
- Regression
- Metric Learning
  - Siamese twin networks
  - Triplet networks

- Test script (takes single model, spits out kaggle-ready CSV or raw output predictions)
- Test script takes output predictions of several models, combines them to spit out csv

Ideas to try:
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

Broad Strategy:
- Train several models
- Average the network predictions over many of these models
- For each image, average the prediction over several crops and orientations of the image (for example at test time, every image's prediction is the average of 4 rotations of the image)
- Win
