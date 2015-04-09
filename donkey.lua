require 'image'
tds=require 'tds'
utils=paths.dofile('utils.lua') -- utils.lua in same directory
torch.setdefaulttensortype('torch.FloatTensor')
local sampleSize = {3, opt.sampleSize, opt.sampleSize}
local imagesRoot = paths.concat(opt.dataRoot, 'train_' .. opt.sampleSize)

local function loadImage(rawJPG)
   local input = image.decompressJPG(rawJPG, 3, 'float')
   local iH = input:size(3)
   local iW = input:size(2)
   -- find the smaller dimension, and resize it to opt.loadSize (while keeping aspect ratio)
   if iH < iW and iH ~= opt.loadSize then
      input = image.scale(input, opt.loadSize, opt.loadSize * iW / iH)
   elseif iW ~= opt.loadSize then
      input = image.scale(input, opt.loadSize * iH / iW, opt.loadSize)
   end
   -- mean/std
   for i=1,3 do -- channels
      if mean then input[{{i},{},{}}]:add(-mean[i]) end
      if std then input[{{i},{},{}}]:div(std[i]) end
   end
   return input
end

-- get random crop
local function processTrain(rawJPG)
   collectgarbage()
   local input = loadImage(rawJPG)
   -- do random crop
   local h1 = math.ceil(torch.uniform(1e-2, input:size(2) - sampleSize[2]))
   local w1 = math.ceil(torch.uniform(1e-2, input:size(3) - sampleSize[3]))
   local out = image.crop(input, w1, h1, w1 + sampleSize[3], h1 + sampleSize[2])
   assert(out:size(2) == sampleSize[2], 'wrong crop size')
   assert(out:size(3) == sampleSize[3], 'wrong crop size')
   -- do flip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end             -- horizontal flip
   if torch.uniform() > 0.5 then out = image.vflip(out); end             -- vertical flip
   -- out = image.rotate(out, torch.uniform() * math.pi * 2, 'bilinear') -- rotation jitter
   return out
end

-- get center crop
local function processTest(rawJPG)
   collectgarbage()
   local input = loadImage(rawJPG)
   local w1 = math.ceil((input:size(3) - sampleSize[3])/2)
   local h1 = math.ceil((input:size(2) - sampleSize[2])/2)
   local out = image.crop(input, w1, h1, w1 + sampleSize[3], h1 + sampleSize[2]) -- center patch
   return out
end

function getTrainingMiniBatch(quantity)
   local data = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[3])
   local label = torch.Tensor(quantity)
   for i=1, quantity do -- class-balanaced sampling
      local class = torch.random(1, 5)
      local index = torch.random(1, #train_data[class])
      local out = processTrain(train_data[class][index])
      data[i]:copy(out)
      label[i] = class
   end
   return data, label
end

function getValidationData(i1, i2)
   local data = torch.Tensor(i2-i1+1, sampleSize[1], sampleSize[2], sampleSize[3])
   local label = torch.Tensor(i2-i1+1)
   for i=i1, i2 do
      local out = processTest(val_paths[i])
      data[i-i1+1]:copy(out)
      label[i-i1+1] = val_labels[i]
   end
   return data, label
end

-------------------------
-- Load training data
-------------------------
-- train data is stored in a simple way. 
-- one table is stored, which has 5 members: 1,2,3,4,5
-- Each of the members is a tds.hash with the list of image jpegs of that class (stored as ByteTensor)
train_data = {}
for i=1,5 do
   train_data[i] = tds.hash() 
end
-- load labels from file
for l in io.lines(paths.concat(opt.dataRoot, 'train_labels.txt')) do
   local path, label = unpack(l:split(','))
   if tonumber(label) then
      label = tonumber(label) + 1 --make it 1-indexed
      train_data[label][#train_data[label]+1]
	 = utils.loadFileAsByteTensor(paths.concat(imagesRoot, path .. '.jpeg'))
   end
end
-- val data is stored even more simpler. everything is in one tds.hash as path,label pairs
val_paths = tds.hash()
val_labels = tds.hash()
for l in io.lines(paths.concat(opt.dataRoot, 'val_labels.txt')) do
   local path, label = unpack(l:split(','))
   if tonumber(label) then
      label = tonumber(label) + 1 --make it 1-indexed
      val_paths[#val_paths+1] 
	 = utils.loadFileAsByteTensor(paths.concat(imagesRoot, path .. '.jpeg'))
      val_labels[#val_labels+1] = label
   end
end

collectgarbage()
-----------------------------------------
-- estimate mean/std per channel
-----------------------------------------
do
   local meanstdCacheFile = 'meanstdCache.t7'
   if paths.filep(meanstdCacheFile) then
      print('Loading mean/std from cache file')
      local meanstd = torch.load(meanstdCacheFile)
      mean = meanstd.mean
      std  = meanstd.std
   else
      print('Estimating mean/std from a few images in dataset. Will be cached for future use.')
      local nSamples = 1000
      local meanEstimate = {0,0,0}
      for i=1,nSamples do
	 local img = getTrainingMiniBatch(1)[1]
	 for j=1,3 do meanEstimate[j] = meanEstimate[j] + img[j]:mean() end
      end
      for j=1,3 do meanEstimate[j] = meanEstimate[j] / nSamples end
      mean = meanEstimate
      local stdEstimate = {0,0,0}
      for i=1,nSamples do
	 local img = getTrainingMiniBatch(1)[1]
	 for j=1,3 do stdEstimate[j] = stdEstimate[j] + img[j]:std() end
      end
      for j=1,3 do stdEstimate[j] = stdEstimate[j] / nSamples end
      std = stdEstimate
      torch.save(meanstdCacheFile, {mean=mean, std=std})
   end
end
