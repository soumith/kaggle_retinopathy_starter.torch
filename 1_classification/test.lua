require 'cunn'
require 'optim'
require 'pl'
require 'cudnn'
require 'image'
local dir = require 'pl.dir'

paths.dofile('../fbcunn_files/AbstractParallel.lua')
paths.dofile('../fbcunn_files/ModelParallel.lua')
paths.dofile('../fbcunn_files/DataParallel.lua')
paths.dofile('../fbcunn_files/Optim.lua')

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
   model='small', -- models/[name].lua will be loaded
   bestAccuracy = 0,
   test='',
   testModel='model_8.t7',
   retrain='',
   loadSize=256, -- height/width of image to load
   sampleSize=240,-- height/width of image to sample
   dataRoot='../data' -- data in current folder
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

utils=paths.dofile('../utils.lua') -- utils.lua in same directory


-------------- load model --------------
diskModel = torch.load(opt.testModel)

model = diskModel.model
--print(diskModel)
model:cuda()

collectgarbage()
print('Loaded Model from model file.')

local files  = dir.getfiles(paths.concat('../data','test'))
local meanstdCache = paths.concat(paths.cwd(),'meanstdCache.t7')

if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
end

function loadImage(rawImage)
    input = image.decompressJPG(rawImage,'3','float')
    iW = input:size(2)
    iH = input:size(3)
    
    if iH < iW and iH ~= tonumber(opt.loadSize) then
        input = image.scale(input, opt.loadSize, opt.loadSize * iW / iH)
    elseif iW ~= tonumber(opt.loadSize) then
        if (opt.loadSize * iH/iW) < opt.sampleSize then
	   input = image.scale(input,opt.loadSize,opt.loadSize)
	else
           input = image.scale(input, opt.loadSize,opt.loadSize * iH / iW)
	end
    end
    for i=1,3 do
    	if mean then input[{{i},{},{}}]:add(-mean[i]) end
	if std then input[{{i},{},{}}]:div(std[i]) end
    end

    w1 = math.ceil(math.abs((input:size(2)-opt.sampleSize))/2)
    h1 = math.ceil(math.abs((input:size(3)-opt.sampleSize))/2)
    return image.crop(input,h1,w1,h1+opt.sampleSize,w1+opt.sampleSize)
end

function getTestingMiniBatch(indexStart,indexEnd)
    local quantity = indexEnd - indexStart + 1
    local data = torch.Tensor(quantity,3,opt.sampleSize,opt.sampleSize)
    local filenames = {}
    for i=1,quantity do
    	local out = loadImage(utils.loadFileAsByteTensor(files[i+indexStart-1]))
	data[i]:copy(out)
	table.insert(filenames,files[i+indexStart-1])
    end
    return data,filenames
end


local numBatches = (#files + opt.batchSize-1)/opt.batchSize
print('numBatches: '..numBatches)

df = torch.DiskFile('predictions','rw')
for batchNumber = 1,numBatches do

    local batchElems = math.min(opt.batchSize, math.abs((batchNumber-1) * opt.batchSize - #files))
    print('Processing batchNumber: '..batchNumber..', batchSize: '..batchElems)

    local indexStart = 1 + (batchNumber-1) * opt.batchSize
    local indexEnd = indexStart + batchElems - 1

    local inputs, filenames = getTestingMiniBatch(indexStart,indexEnd)
   
    local inputGPU = torch.CudaTensor()
    inputGPU:resize(inputs:size()):copy(inputs)

    outputBatch = torch.exp(model:forward(inputGPU))
    predictions = {}
    local time = sys.clock()    
    for j=1,outputBatch:size(1) do
    	local i = 1
    	local output = outputBatch[j]
	if paths.basename(filenames[j]) == '120_left.jpeg' then print(filenames[j]) end
    	output:apply(function(x) if x == output:max() then df:writeString(paths.basename(filenames[j])..','..i-1) df:writeString('\n') end i=i+1 end)
    end
    
    time = (sys.clock() - time)/outputBatch:size(1)
    print('Time per test example: '..time)
    collectgarbage()
end
df:close()
