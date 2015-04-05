local utils = {}
function utils.loadFileAsByteTensor(path)
   local f = torch.DiskFile(path, 'r'):binary()
   f:seekEnd()
   local size = f:position()
   f:seek(1)
   local out = f:readByte(size-1)
   f:close()
   return torch.ByteTensor(out)
end

-- used for classification task
function utils.get_top1(outputs, labels)
   outputs = outputs:float()
   local top1 = 0
   local _,out_sorted = outputs:sort(2, true) -- descending
   for i=1,labels:size(1) do
      if out_sorted[i][1] == labels[i] then top1 = top1 + 1; end
   end
   return top1
end

function utils.cleanup(net)
   local list = net:listModules()
   for _,val in ipairs(list) do
      for name,field in pairs(val) do
         if torch.type(field) == 'cdata' then val[name] = nil end
         if name == 'homeGradBuffers' then val[name] = nil end
         if name == 'input_gpu' then val['input_gpu'] = {} end
         if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
         if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
         if (name == 'output' or name == 'gradInput') then
            val[name] = field.new()
         end
         if  name == 'buffer' or name == 'buffer2' or name == 'normalized'
         or name == 'centered' or name == 'addBuffer' then
            val[name] = nil
         end
      end
   end
   return net
end

return utils
