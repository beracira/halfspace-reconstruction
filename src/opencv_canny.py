import cv2 as cv
from image_gen import Image_Generator
from collections import deque
import numpy as np
from PIL import Image

image_size = 512

# ig = Image_Generator(size=image_size, mode='curve', 
#     noisy=True, blur=True)
# original, filtered = ig.get_new_image_pair()
image = cv.imread('result/1ji.png', cv.IMREAD_GRAYSCALE)

edges = cv.Canny(image, 100, 200)

img = Image.fromarray(edges, 'L')
img.save("result/canny.png")
