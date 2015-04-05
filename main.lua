require 'cunn'
require 'optim'
paths.dofile('fbcunn_files/AbstractParallel.lua')
paths.dofile('fbcunn_files/ModelParallel.lua')
paths.dofile('fbcunn_files/DataParallel.lua')
paths.dofile('fbcunn_files/Optim.lua')

opt = {
   epoch=1,
   learningRate = 0.1,
   decay = 0.2,
   weightDecay = 5e-4,
   momentum = 0.9,
   manualSeed = 1,
   nDonkeys = 4,
   nEpochs = 30,
   batchSize = 128,
   GPU = 1,
   nGPU = 4,
   epochSize = 10000,
   model='alexnetowtbn', -- models/[name].lua will be loaded
   bestAccuracy = 0,
   retrain='',
   loadSize=256, -- height/width of image to load
   sampleSize=224,-- height/width of image to sample
   dataRoot='./data' -- data in current folder
}
-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)
include('data.lua')
utils=require('utils') -- utils.lua in same directory

-------------- create model --------------
if opt.retrain ~= '' then -- load model from disk for retraining
   model = torch.load(opt.retrain).model
else
   local config = opt.model
   paths.dofile('models/' .. config .. '.lua')
   print('=> Creating model from file: models/' .. config .. '.lua')
   model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
end
criterion = nn.ClassNLLCriterion()
print(model)
model:cuda()
criterion:cuda()
collectgarbage()

-- GPU inputs (preallocate)
inputs = torch.CudaTensor()
labels = torch.CudaTensor()
local timer = torch.Timer()
local dataTimer = torch.Timer()
-------------------- training functions ------------------------
function train()
   print("==> Training epoch # " .. opt.epoch)
   top1=0; loss = 0; batchNumber = 0
   model:training()
   local timer = torch.Timer()
   for i=1,opt.epochSize do
      donkeys:addjob(function() return getTrainingMiniBatch(opt.batchSize) end, trainBatch)
   end
   donkeys:synchronize()
   cutorch.synchronize()

   top1 = top1 * 100 / (opt.batchSize * opt.epochSize)
   loss = loss / opt.epochSize
   total_time = timer:time().real
   train_loss = loss
   train_accuracy = top1
end

function trainBatch(inputsCPU, labelsCPU)
   local dataLoadingTime = dataTimer:time().real; timer:reset(); -- timers
   batchNumber = batchNumber + 1
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, outputs = optimizer:optimize(optim.sgd, inputs, labels, criterion)
   loss = loss + err
   top1 = top1 + utils.get_top1(outputs, labelsCPU)
   print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f Err %.4f '):format(
         opt.epoch, batchNumber, opt.epochSize, timer:time().real, dataLoadingTime, err))
   cutorch.synchronize(); collectgarbage();
   dataTimer:reset()
end
-------------------- testing functions ------------------------
function test()
   print("==> Validation epoch # " .. opt.epoch)
   top1 = 0; loss = 0; batchNumber = 0
   model:evaluate()
   local timer = torch.Timer()
   for i=1,nTest/opt.batchSize do -- nTest is set in data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(function() return getValidationData(indexStart, indexEnd) end, testBatch)
   end
   donkeys:synchronize()
   cutorch.synchronize()
   top1 = top1 * 100 / nTest
   loss = loss / (nTest/opt.batchSize)
   print({epoch = opt.epoch,
	  train_time = total_time,
	  train_loss = train_loss,
	  train_accuracy = train_accuracy,
	  test_time = timer:time().real,
	  test_loss = loss,
	  test_accuracy = top1,
	  best_accuracy = opt.bestAccuracy})
   return top1
end

function testBatch(inputsCPU, labelsCPU)
   cutorch.synchronize(); collectgarbage();
   batchNumber = batchNumber + opt.batchSize
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, labels)
   loss = loss + err
   top1 = top1 + utils.get_top1(outputs, labelsCPU)
end

-----------------------------------------------------------------------------
local accs = {}
while (opt.epoch < opt.nEpochs) do
   optimizer = nn.Optim(model, opt)
   train()
   local acc = test()
   accs[#accs+1] = acc
   if (#accs > 3) and (accs[#accs] < accs[#accs - 1] * 1.01)
      and (accs[#accs] < accs[#accs - 2] * 1.01) then
         opt.learningRate = opt.learningRate * opt.decay
         if opt.learningRate < 1e-4 then
            print('stopping job'); os.exit(0)
         end
   end
   if acc > opt.bestAccuracy then
      opt.bestAccuracy = acc
      torch.save('model_' .. opt.epoch .. '.t7',
                 {model=utils.cleanup(model), opt=opt})
   end
   opt.epoch = opt.epoch + 1
end
