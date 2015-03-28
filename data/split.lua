assert(arg[1])
require 'xlua'
f = arg[1]
labels = {}
for l in io.lines(f) do
   labels[#labels+1] = l
end
local tr = assert(io.open('train_labels.txt', 'w'))
local vl = assert(io.open('val_labels.txt', 'w'))
c=0
for l in io.lines('trainLabels.csv') do
   c=c+1
   local valsample = false
   for i=1,#labels do
      if l:split('_')[1] == labels[i] then
	 -- print(l, labels[i])
	 valsample = true
	 break
      end
   end
   if valsample then
      vl:write(l .. '\n')
   else
      tr:write(l .. '\n')
   end
   xlua.progress(c, 35000)
   if math.random(1,4) == 3 then
      collectgarbage()
   end
end

tr:close()
vl:close()
print('Finished splitting labels to train/val')
