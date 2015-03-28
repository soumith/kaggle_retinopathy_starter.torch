require 'cunn'
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
   nDonkeys = 2,
   nEpochs = 30,
   batchSize = 128,
   GPU = 1,
   nGPU = 4,
   backend = 'cudnn',
   epochSize = 10000,
   model='alexnetowtbn',
   bestAccuracy = 0,
   retrain='',
   dataRoot='./data' -- data in current folder
}
for k,v in pairs(opt) do opt[k] = os.getenv(k) or opt[k] end
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)
include('data.lua')
include('utils.lua')

-------------- create model --------------
if opt.retrain ~= '' then -- load model from disk for retraining
   model = torch.load(opt.retrain).model
else
   local config = opt.model .. '_' .. opt.backend
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
   loss = 0; batchNumber = 0
   model:training()
   local timer = torch.Timer()
   for i=1,opt.epochSize do
      donkeys:addjob(function() return trainLoader:sample(opt.batchSize) end, trainBatch)
   end
   donkeys:synchronize()
   cutorch.synchronize()

   loss = loss / opt.epochSize
   total_time = timer:time().real
   train_loss = loss
   train_accuracy = 0 -- TODO
end

function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize(); collectgarbage();
   batchNumber = batchNumber + 1
   local dataLoadingTime = dataTimer:time().real; timer:reset(); -- timers
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, outputs = optimizer:optimize(optim.sgd, inputs, labels, criterion)
   loss = loss + err
   print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f Err %.4f '):format(
         opt.epoch, batchNumber, opt.epochSize, timer:time().real, dataLoadingTime, err))
   dataTimer:reset()
end
-------------------- testing functions ------------------------
function test()
   print("==> Validation epoch # " .. opt.epoch)
   loss = 0; batchNumber = 0
   model:evaluate()
   local timer = torch.Timer()
   for i=1,nTest/opt.batchSize do -- nTest is set in data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(function() return testLoader:get(indexStart, indexEnd) end, testBatch)
   end
   donkeys:synchronize()
   cutorch.synchronize()

   loss = loss / (nTest/opt.batchSize)
   print({epoch = opt.epoch,
	  train_time = total_time,
	  train_loss = train_loss,
	  train_accuracy = train_accuracy,
	  test_time = timer:time().real,
	  test_loss = loss,
	  test_accuracy = 0, -- TODO
	  best_accuracy = opt.bestAccuracy})
   return 0 -- TODO
end

function testBatch(inputsCPU, labelsCPU)
   cutorch.synchronize(); collectgarbage();
   batchNumber = batchNumber + opt.batchSize
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, labels)
   loss = loss + err
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
            print('stopping job'); os.exit()
         end
   end
   if acc > opt.bestAccuracy then
      opt.bestAccuracy = acc
      torch.save('model_' .. opt.epoch .. '.t7',
                 {model=sanitize(model), opt=opt})
   end
   opt.epoch = opt.epoch + 1
end
