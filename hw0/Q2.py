import sys
from PIL import Image
input_file = str(sys.argv[1])
im = Image.open(input_file)
pix = im.load()
r, g, b = im.split()
rpix = r.load()
gpix = g.load()
bpix = b.load()
#im2 = Image.new(im.mode, im.size)
for i in range(im.size[0]):    # for every col:
    for j in range(im.size[1]):    # For every row
        pix[i,j] = (int(rpix[i,j]/2), int(gpix[i,j]/2), int(bpix[i,j]/2)) # set the colour accordingly
        im.putpixel((i,j),pix[i,j])
im.save('Q2.png')
