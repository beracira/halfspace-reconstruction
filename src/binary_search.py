import cv2 as cv
from image_gen import Image_Generator
from collections import deque
import numpy as np
from PIL import Image

image_size = 512

ig = Image_Generator(size=image_size, mode='curve', 
    noisy=True, blur=True)
original, filtered = ig.get_new_image_pair()
print (original)
img = Image.fromarray(original, 'L')
img.save("result/original.png")

def find_change_point(line):
    counter = 0
    i = 0
    j = line.shape[0] - 1
    while counter < 20 and i < j:
        mid = (i + j) // 2
        if line[mid] != line[mid + 1]:
            return mid
        elif line[i] == line[mid]:
            i = mid + 1
        else:
            j = mid
    return (i + j) // 2

def is_on_line(p1, p2, target):
    d = np.linalg.norm(np.cross(p2 - p1, p1 - target))/np.linalg.norm(p2 - p1)
    # print (d)
    if d < 1:
        return 1
    return 0

def find_halfspace(img):
    """
    returns an image with black background and boundary marked in white dots
    """
    height, width = img.shape
    if height <= 1: return img.copy()

    top_left = img[0, 0]
    top_right = img[0, -1]
    bot_left = img[-1, 0]
    bot_right = img[-1, -1]

    num_black_pts = int(sum((top_left, top_right, bot_left, bot_right)))
    print ("num_black_pts:", num_black_pts)
    if num_black_pts == 0 or num_black_pts == 4:
        # recurse here
        pass
    elif num_black_pts == 1 or num_black_pts == 3:
        # find boundary
        # print (img)
        if num_black_pts == 3:
            img = np.where((img==0)|(img==1), img^1, img)
            top_left = img[0, 0]
            top_right = img[0, -1]
            bot_left = img[-1, 0]
            bot_right = img[-1, -1]
        first, second = None, None
        if top_left == 1:
            print("checking top left")
            first = (0, find_change_point(img[0, :]))
            second = (find_change_point(img[:, 0]), 0)
        elif top_right == 1:
            first = (0, find_change_point(img[0, :]))
            second = (find_change_point(img[:, -1]), width - 1)
        elif bot_left == 1:
            first = (find_change_point(img[:, 0]), 0)
            second = (height - 1, find_change_point(img[-1, :]))
        else:
            first = (find_change_point(img[:, -1]), width - 1)
            second = (height - 1, find_change_point(img[-1, :]))
        retval = np.zeros((height, width), dtype=np.uint8)
        first, second = np.array(first), np.array(second)
        print (first, second)
        for i in range(height):
            for j in range(width):
                retval[i, j] = is_on_line(first, second, np.array((i, j)))
                    
        return retval
    else:
        pass

img = np.tri(N=512, M=512, k=2, dtype=np.uint8)
img = np.fliplr(img)
# img *= 255
print (img)
edges = find_halfspace(img)
edges *= 255
print (edges)

img = Image.fromarray(edges, 'L')
img.save("result/binary_search.png")
