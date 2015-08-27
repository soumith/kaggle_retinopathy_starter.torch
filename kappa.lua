require 'nn'
require 'image'
tds = require 'tds'
require 'threads'

function quad_kappa(actual, pred)

	local num_ratings = actual:size(2)
	local batch_size = actual:size(1)

	local x = torch.Tensor(num_ratings,1)
	local i = 0
	local y = x:apply(function() i=i+1 return i end):repeatTensor(1,num_ratings)
	local a = (y-y:t())
	local w = a:cmul(a):div((num_ratings - 1) * (num_ratings - 1))

	torch.manualSeed(1)

	local hist_ratings_a = actual:sum(1)
	local hist_ratings_b = pred:sum(1)

	local M = torch.Tensor(5,5):zero()
	M:addmm(actual:t(),pred)
	local numerator = torch.sum(w:cmul(M))

	local D = torch.Tensor(num_ratings,num_ratings):zero()
	D:addmm(hist_ratings_a:resize(num_ratings,1),hist_ratings_b)
	local denom = torch.sum(w:cmul(D))

	return 1 - (numerator/denom)

end

function convert_to_one_hot_vector_rep(batch_size, out, indices)
    for i=1,batch_size do
        out[i][indices[i]] = 1
    end
end

function random_indices(num_indices, low, high) 
    local indices = torch.Tensor(num_indices)
    indices:apply(function() return math.floor(low + (high - low) * torch.uniform()) end)
    return indices
end

local num_ratings = 5
local batch_size=128
-- for testing.
local indices = random_indices(128, 1, 6)
local actual = torch.Tensor(batch_size,num_ratings):zero()
convert_to_one_hot_vector_rep(batch_size,actual, indices)

local pred = torch.Tensor(batch_size,num_ratings):zero()
local pred_indices = random_indices(128, 1, 6)
convert_to_one_hot_vector_rep(batch_size,pred,pred_indices)

print("Kappa: "..quad_kappa(actual, pred))

