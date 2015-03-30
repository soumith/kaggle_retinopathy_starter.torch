function loadFileAsByteTensor(path)
   local f = torch.DiskFile(path, 'r'):binary()
   f:seekEnd()
   local size = f:position()
   f:seek(1)
   local out = f:readByte(size-1)
   f:close()
   return torch.ByteTensor(out)
end
