import cv2 as cv
from image_gen import Image_Generator
from collections import deque
import numpy as np
from PIL import Image
from timeit import default_timer as timer

image_size = 512

# ig = Image_Generator(size=image_size, mode='curve', 
#     noisy=True, blur=True)
# original, filtered = ig.get_new_image_pair()
image = cv.imread('result/1ji.png', cv.IMREAD_GRAYSCALE)

start = timer()
edges = cv.Canny(image, 100, 200)
end = timer()
print (end - start)

img = Image.fromarray(edges, 'L')
img.save("result/canny.png")
