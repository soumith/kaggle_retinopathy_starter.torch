function createModel(nGPU)
   require 'cudnn'

   local features = nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,32,7,7))
   features:add(nn.SpatialBatchNormalization(32,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(nn.SpatialLPPooling(32,2,3,3,2,2))  
   features:add(cudnn.SpatialConvolution(32,48,5,5,1,1,2,2))
   features:add(nn.SpatialBatchNormalization(48,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(nn.SpatialLPPooling(48,2,3,3,2,2))           
   features:add(cudnn.SpatialConvolution(48,64,3,3,1,1,1,1))
   features:add(nn.SpatialBatchNormalization(64,1e-3))
   features:add(cudnn.ReLU(true))
   
   local classifier = nn.Sequential()
   classifier:add(nn.View(64*11*11))
   classifier:add(nn.Linear(64*11*11, 256))
   classifier:add(nn.BatchNormalization(256))
   classifier:add(nn.ReLU())
   classifier:add(nn.Linear(256, nClasses))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential():add(features):add(classifier)

   return model
end
