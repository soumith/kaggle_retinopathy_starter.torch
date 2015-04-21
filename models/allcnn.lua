function createModel(nGPU)
   require 'cudnn'

   convnet = nn.Sequential()
   convnet:add(nn.SpatialConvolution(1,96,3,3))
   convnet:add(nn.ReLU())
   convnet:add(nn.SpatialConvolution(96,96,3,3))
   convnet:add(nn.ReLU())
   convnet:add(nn.SpatialConvolution(96,192,3,3,2,2))
   convnet:add(nn.ReLU())
   convnet:add(nn.SpatialConvolution(192,192,3,3))
   convnet:add(nn.ReLU())
   convnet:add(nn.SpatialConvolution(192,192,3,3))
   convnet:add(nn.ReLU())
   convnet:add(nn.SpatialConvolution(192,192,3,3,2,2))
   convnet:add(nn.ReLU())
   convnet:add(nn.SpatialMaxPooling(3,3,2,2))
   convnet:add(nn.SpatialConvolution(192,192,3,3))
   convnet:add(nn.ReLU())
   convnet:add(nn.SpatialConvolution(192,192,1,1))
   convnet:add(nn.ReLU())
   convnet:add(nn.SpatialConvolution(192,10,1,1))
   convnet:add(nn.ReLU())
   convnet:add(nn.View(10*27*27))
   convnet:add(nn.Linear(10*27*27,10))
   convnet:add(nn.ReLU())
   convnet:add(nn.Linear(10,nClasses))
   convnet:add(nn.LogSoftMax())

   return convnet 
end

