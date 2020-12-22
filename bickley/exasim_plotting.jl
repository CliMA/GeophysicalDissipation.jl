using Images, FileIO, ImageIO
using QuartzImageIO, ImageMagick

filename = "128sq_p2_low"
searchdir(path, key) = filter(x -> occursin(key, x), readdir(path))
pngfiles = searchdir(pwd(), filename)

# to make gif
p = load(pwd() * "/" * pngfiles[1])
tmp = [load(pwd() * "/" * pngfiles[i]) for i in eachindex(pngfiles)]
size(tmp)
tmp2 = reduce(hcat, tmp)
tmp3 = reshape(tmp2, (size(tmp[1])...,length(tmp)))
save(filename * "_exasim.gif", tmp3)