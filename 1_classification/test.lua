require 'cunn'
require 'optim'
require 'pl'
require 'cudnn'
require 'image'
local tds = require 'tds'
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
   testModel='results/l2pooling/model_29.t7',
   retrain='',
   loadSize=256, -- height/width of image to load
   sampleSize=224,-- height/width of image to sample
   dataRoot='../data', -- data in current folder
   validation=''
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

--local files  = dir.getfiles(paths.concat(opt.dataRoot,'test'))

function get_validation_set()
    local val_data = tds.hash()
    local val_data_labels = tds.hash()

    for line in io.lines(paths.concat(opt.dataRoot, 'val_labels.txt')) do
    	local filename,label = unpack(line:split(','))
    	label = tonumber(label)
    	val_data[#val_data+1] = paths.concat(paths.concat(opt.dataRoot,'train'),filename..'.jpeg')
    	val_data_row = tds.hash()
    	val_data_row['name'] = filename
    	val_data_row['label'] = label
    	val_data_labels[#val_data_labels+1] = val_data_row
    end
    return val_data, val_data_labels
end

if opt.validation ~='' then
   files, val_data_labels = get_validation_set()
else
   files = dir.getfiles(paths.concat(opt.dataRoot,'test'))
   print("num files: "..#files)
end

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

local predictions = tds.hash()
local totalTime = sys.clock()

for batchNumber = 1,numBatches do

    local batchElems = math.min(opt.batchSize, math.abs((batchNumber-1) * opt.batchSize - #files))
    print('Processing batchNumber: '..batchNumber..', batchSize: '..batchElems)

    local indexStart = 1 + (batchNumber-1) * opt.batchSize
    local indexEnd = indexStart + batchElems - 1

    local inputs, filenames = getTestingMiniBatch(indexStart,indexEnd)
   
    local inputGPU = torch.CudaTensor()
    inputGPU:resize(inputs:size()):copy(inputs)

    outputBatch = torch.exp(model:forward(inputGPU))

    local time = sys.clock()    
    for j=1,outputBatch:size(1) do
    	local i = 1
    	local output = outputBatch[j]
	local filename = unpack(paths.basename(filenames[j]):split('.jpeg'))
    	output:apply(function(x) if x == output:max() then prediction=tds.hash() prediction['name']=filename prediction['label']=i-1 predictions[#predictions+1]=prediction end i=i+1 end)
    end

    time = (sys.clock() - time)/outputBatch:size(1)
    print('Time per test example: '..time)
    collectgarbage()
end
totalTime = sys.clock() - totalTime
print('Total Time taken including data loading and prep: '..totalTime)

local counter = 0
local err = 0

local df = torch.DiskFile('predictions','rw')
df:writeString('image,level')
df:writeString('\n')
for i=1,#predictions do
    pFilename = predictions[i]['name']
    pLabel = predictions[i]['label']
    df:writeString(pFilename..','..pLabel)
    df:writeString('\n')

    if opt.validation ~= '' then
        vFilename = val_data_labels[i]['name']
   	vLabel = val_data_labels[i]['label']
	if pFilename == vFilename then counter = counter + 1 end
	if pLabel ~= vLabel then err = err + 1 end
    end
end
df:close()

if opt.validation ~= '' then
   print('Total Predictions: '..#predictions..', counter: '..counter..', err: '..err)
   local accuracy = (tonumber(counter) - tonumber(err)) / tonumber(counter)
   print('Model Accuracy on Validation Set: '..accuracy)
end

