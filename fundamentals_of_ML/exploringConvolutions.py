import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

print("Hello world")

i = misc.ascent()


# Show image
""" plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show() """

#get image dimensions from its numpy array copy so we can loop over it later
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

#lets create a 3x3 array filter
# This filter detects edges nicely
# It creates a convolution that only passes through sharp edges and straight
# lines.

#Experiment with different values for fun effects.
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
# A couple more filters to try for fun!
#filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
#filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
filter = [[1, 1, 1], [-1, -1, -1], [0, 0, 0]]
# If all the digits in the filter don't add up to 0 or 1, you 
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight  = 1

for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        convolution = 0.0
        convolution = convolution + (i[x - 1, y-1] * filter[0][0])
        convolution = convolution + (i[x, y-1] * filter[1][0])
        convolution = convolution + (i[x + 1, y-1] * filter[2][0])
        convolution = convolution + (i[x-1, y] * filter[0][1])
        convolution = convolution + (i[x, y] * filter[1][1])
        convolution = convolution + (i[x+1, y] * filter[2][1])
        convolution = convolution + (i[x-1, y+1] * filter[0][2])
        convolution = convolution + (i[x, y+1] * filter[1][2])
        convolution = convolution + (i[x+1, y+1] * filter[2][2])
        convolution = convolution * weight

        if(convolution < 0):
            convolution=0
        if(convolution>255):
            convolution=255
        i_transformed[x, y] = convolution

#plot the image to see the effect of the convolution (applied filter)

plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
plt.show()

# POOLING ---------------------------------> HERE! Last step of the pooling step

new_x = int(size_x/4)
new_y = int(size_y/4)
newImage = np.zeros((new_x, )) 
