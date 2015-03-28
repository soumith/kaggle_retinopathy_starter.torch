require 'image'
local ffi=require 'ffi'
local loadSize   = {3, 256, 256}
local sampleSize = {3, 224, 224}

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to 256 (while keeping aspect ratio)
   if input:size(3) < input:size(2) then
      input = image.scale(input, 256, 256 * input:size(2) / input:size(3))
   else
      input = image.scale(input, 256 * input:size(3) / input:size(2), 256)
   end
   -- mean/std
   for i=1,3 do -- channels
      if mean then input[{{i},{},{}}]:add(-mean[i]) end
      if std then input[{{i},{},{}}]:div(std[i]) end
   end
   return input
end

-- get random crop
local function processTrain(path)
   collectgarbage()
   local input = loadImage(path)
   -- do random crop
   local h1 = math.ceil(torch.uniform(1e-2, input:size(2) - sampleSize[2]))
   local w1 = math.ceil(torch.uniform(1e-2, input:size(3) - sampleSize[3]))
   local out = image.crop(input, w1, h1, w1 + sampleSize[3], h1 + sampleSize[2])
   assert(out:size(2) == sampleSize[3], 'wrong crop size')
   assert(out:size(3) == sampleSize[2], 'wrong crop size')
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
   return out
end

-- get center crop
local function processTest(path)
   collectgarbage()
   local input = loadImage(path)
   local w1 = math.ceil((input:size(3) - sampleSize[3])/2)
   local h1 = math.ceil((input:size(2) - sampleSize[2])/2)
   local out = image.crop(input, w1, h1, w1 + sampleSize[3], h1 + sampleSize[2]) -- center patch
   return out
end

local dataset = torch.class('dataLoader')

function dataset:sample(quantity)
   local data = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[3])
   local label = torch.Tensor(quantity)
   for i=1, quantity do
      local class = torch.random(1, #self.classes)
      local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
      local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index] ]))
      local out = processTrain(imgpath)
      data[i]:copy(out)
      label[i] = class
   end
   return data, label
end

function dataset:get(i1, i2)
   local data = torch.Tensor(i2-i1+1, sampleSize[1], sampleSize[2], sampleSize[3])
   local label = torch.Tensor(i2-i1+1)
   for i=1, i2-i1+1 do
      local imgpath = ffi.string(torch.data(self.imagePath[i]))
      local out = processTest(imgpath)
      data[i]:copy(out)
      label[i] = self.imageClass[i]
   end
   return data, label
end

trainLoader = torch.load(trainCache)
testLoader = torch.load(testCache)
mean = meanstd.mean
std = meanstd.std
classes = trainLoader.classes
collectgarbage()
