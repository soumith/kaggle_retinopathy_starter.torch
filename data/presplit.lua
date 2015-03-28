fns = {'ids_4.txt','ids_3.txt','ids_2.txt','ids_1.txt','ids_0.txt'}
per = 0.2 -- percentage
val = {}
math.randomseed(10)

for kkk=1,#fns do
   local fn = fns[kkk]
   local ids = {}
   for l in io.lines(fn) do
      local id = tonumber(l)
      ids[#ids+1] = id
   end

   local nval = math.ceil(per * #ids)

   local val_ids = {}

   for i=1,nval do
      local index = math.random(1,#ids)
      while val_ids[index] do index = math.random(1,#ids) end
      val_ids[index] = true
   end

   for i=1,#ids do
      if val_ids[i] then
	 val[ids[i]] = true
      end
   end
end

for k,v in pairs(val) do
   -- os.execute('cat trainLabels.csv | grep ' .. k)
   print(k)
end
