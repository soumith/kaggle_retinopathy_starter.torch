function createModel(nGPU)
   require 'cudnn'

   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   local features = nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,64,8,8,4,4,1,1))       -- 224 -> 55
   features:add(nn.SpatialBatchNormalization(64,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27

   features:add(cudnn.SpatialConvolution(64,64,5,5,1,1,2,2))       --  27 -> 27
   features:add(nn.SpatialBatchNormalization(64,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   
   --features:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))      --  13 ->  13
   --features:add(nn.SpatialBatchNormalization(64,1e-3))
   --features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
   features:add(nn.Dropout(0.5))
   --features:add(cudnn.ReLU(true))

   if nGPU > 1 then
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local features_single = features
      features = nn.DataParallel(1)
      for i=1,nGPU do
         cutorch.withDevice(i, function()
                               features:add(features_single:clone())
         end)
      end
      features.gradInput = nil
   end

   local classifier = nn.Sequential()
   classifier:add(nn.View(64*13*13))

   local branch1
   if nGPU == 1 then
      branch1 = nn.Concat(2)
   else
      branch1 = nn.ModelParallel(2)
   end
   for i=1,nGPU do
      local s = nn.Sequential()
      s:add(nn.Linear(64*13*13, 4096/nGPU))
      s:add(nn.BatchNormalization(4096/nGPU,1e-3))
      s:add(nn.Dropout(0.5))
      s:add(nn.ReLU())
      branch1:add(s)
   end
   classifier:add(branch1)
   local branch2
   if nGPU == 1 then
      branch2 = nn.Concat(2)
   else
      branch2 = nn.ModelParallel(2)
   end
   for i=1,nGPU do
      local s = nn.Sequential()
      s:add(nn.Linear(4096, 4096/nGPU))
      s:add(nn.BatchNormalization(4096/nGPU,1e-3))
      s:add(nn.Dropout(0.5))
      s:add(nn.ReLU())
      branch2:add(s)
   end
   classifier:add(branch2)
   classifier:add(nn.Linear(4096, nClasses))

   local model = nn.Sequential():add(features):add(classifier)

   return model
end

