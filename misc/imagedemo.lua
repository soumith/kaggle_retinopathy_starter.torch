require 'image'
a=image.load('data/train/10010_right.jpeg')
a=image.scale(a, 384, a:size(2) * 384 / a:size(3))
t = {}
t['original'] = a
t['lab'] = image.rgb2lab(a:clone())
t['yuv'] = image.rgb2yuv(a:clone())
t['hsl'] = image.rgb2hsl(a:clone())
t['hsv'] = image.rgb2hsv(a:clone())
t['polar'] = image.polar(a:clone(), 'bilinear', 'valid')
t['logpolar'] = image.logpolar(a:clone(), 'bilinear', 'valid')

for k,v in pairs(t) do
   image.display{image=v, legend=k}
end
