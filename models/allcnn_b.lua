function createModel(nGPU)
   require 'cudnn'

   local baseModel = nn.Sequential()
   baseModel:add(cudnn.SpatialConvolution(3,96,5,5))
   baseModel:add(cudnn.ReLU(true))
   baseModel:add(cudnn.SpatialConvolution(96,96,1,1))
   baseModel:add(cudnn.ReLU(true))
   baseModel:add(cudnn.SpatialConvolution(96,96,3,3,2,2))
   baseModel:add(cudnn.ReLU(true))
   baseModel:add(cudnn.SpatialConvolution(96,192,5,5))
   baseModel:add(cudnn.ReLU(true))
   baseModel:add(cudnn.SpatialConvolution(192,192,1,1))
   baseModel:add(cudnn.ReLU(true))
   baseModel:add(cudnn.SpatialConvolution(192,192,3,3,2,2))
   baseModel:add(cudnn.ReLU(true))
   
   local higherLayer = nn.Sequential()
   higherLayer:add(cudnn.SpatialConvolution(192,192,3,3))   
   higherLayer:add(cudnn.ReLU(true))                        
   higherLayer:add(cudnn.SpatialConvolution(192,192,1,1))   
   higherLayer:add(cudnn.ReLU(true))                        
   higherLayer:add(cudnn.SpatialConvolution(192,10,1,1))    
   higherLayer:add(cudnn.ReLU(true))                        
   higherLayer:add(nn.View(10*24*24))                       
   higherLayer:add(nn.Linear(10*24*24,10))                  
   higherLayer:add(nn.ReLU())                               
   higherLayer:add(nn.Linear(10,nClasses))                  
   higherLayer:add(nn.LogSoftMax())
   
   local model = nn.Sequential():add(baseModel):add(higherLayer)
   return model
end
