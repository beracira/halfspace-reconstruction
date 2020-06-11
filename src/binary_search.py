import cv2 as cv
from image_gen import Image_Generator
from collections import deque
import numpy as np
from PIL import Image

pixel_counter = 0
counter_arr = None

def similar_color(x, y):
    return abs(x - y) > 32

def find_change_point(line):
    counter = 0
    i = 0
    j = line.shape[0] - 1
    while counter < 20 and i < j:
        global pixel_counter
        pixel_counter += 2
        mid = (i + j) // 2
        if not similar_color(line[mid], line[mid + 1]):
            return mid
        elif similar_color(line[i], line[mid]):
            i = mid + 1
        else:
            j = mid
    return (i + j) // 2

def is_on_line(p1, p2, target):
    d = np.linalg.norm(np.cross(p2 - p1, p1 - target))/np.linalg.norm(p2 - p1)
    if d < 1:
        return 1
    return 0

def get_black_pts(img, x, y):
    height, width = img.shape
    top_left = img[0, 0]
    top_right = img[0, -1]
    bot_left = img[-1, 0]
    bot_right = img[-1, -1]
    global counter_arr
    counter_arr[x, y] += 1
    counter_arr[x + height - 1, y] += 1
    counter_arr[x, y + width - 1] += 1
    counter_arr[x + height - 1, y + width - 1] += 1  

    corner_arr = np.array((top_left, top_right, bot_left, bot_right))
    avg = np.mean(corner_arr)
    return *(corner_arr > avg * 1.1), corner_arr[corner_arr > avg * 1.1].shape[0]


def recursion_helper(retval, img, x, y):
    height, width = img.shape
    retval[0:height // 2, 0:width // 2] = find_halfspace(img[0:height // 2, 0:width // 2], x, y)
    retval[height // 2:, 0:width // 2] = find_halfspace(img[height // 2:, 0:width // 2], x + height // 2, y)
    retval[0:height // 2, width // 2:] = find_halfspace(img[0:height // 2, width // 2:], x, y + width // 2)
    retval[height // 2:, width // 2:] = find_halfspace(img[height // 2:, width // 2:], x + height // 2, y + width // 2)

def find_halfspace(img, x, y):
    """
    returns an image with black background and boundary marked in white dots
    """

    height, width = img.shape
    retval = np.zeros((height, width), dtype=np.uint8)
    if height >= 32:
        recursion_helper(retval, img, x, y)
        return retval

    if height <= 1: 
        return retval

    top_left, top_right, bot_left, bot_right, num_black_pts = get_black_pts(img, x, y)

    if num_black_pts == 0 or num_black_pts == 4:
        recursion_helper(retval, img, x, y)
    elif num_black_pts == 1 or num_black_pts == 3:
        # find boundary
        # print (img)
        if num_black_pts == 3:
            top_left = not top_left
            top_right = not top_right
            bot_left = not bot_left
            bot_right = not bot_right
        first, second = None, None
        if top_left == True:
            first = (0, find_change_point(img[0, :]))
            second = (find_change_point(img[:, 0]), 0)
        elif top_right == True:
            first = (0, find_change_point(img[0, :]))
            second = (find_change_point(img[:, -1]), width - 1)
        elif bot_left == True:
            first = (find_change_point(img[:, 0]), 0)
            second = (height - 1, find_change_point(img[-1, :]))
        else:
            first = (find_change_point(img[:, -1]), width - 1)
            second = (height - 1, find_change_point(img[-1, :]))
        first, second = np.array(first), np.array(second)
        for i in range(height):
            for j in range(width):
                retval[i, j] = is_on_line(first, second, np.array((i, j)))
    else:
        if top_left == bot_right or bot_left == top_right:
            recursion_helper(retval, img, x, y)
        else:
            if top_left == top_right:
                first = (find_change_point(img[:, 0]), 0)
                second = (find_change_point(img[:, -1]), width - 1)
            elif top_right == bot_right:
                first = (0, find_change_point(img[0, :]))
                second = (height - 1, find_change_point(img[-1, :]))
            else:
                print ("something went wrong...")
            first, second = np.array(first), np.array(second)
            for i in range(height):
                for j in range(width):
                    retval[i, j] = is_on_line(first, second, np.array((i, j)))

    return retval

# image_size = 512

print ("Generating testing image pair...")
image = cv.imread('result/1ji.png', cv.IMREAD_GRAYSCALE)
# print (image)

# ig = Image_Generator(size=image_size, mode='curve', 
#     noisy=True, blur=True)
# original, filtered = ig.get_new_image_pair()
# img = Image.fromarray(original, 'L')
# img.save("result/original.png")

# img = np.tri(N=512, M=512, k=2, dtype=np.uint8)
# img = np.fliplr(img)
# img *= 255
# print (img)
# print (filtered)
print ("Finding boundary on testing image...")
if counter_arr is None:
    counter_arr = np.zeros(image.shape, dtype=np.uint8)
edges = find_halfspace(image, 0, 0)
edges *= 255
# print (edges)

img = Image.fromarray(edges, 'L')
img.save("result/binary_search.png")

print ("Total number of pixels seen:", np.sum(counter_arr) + pixel_counter)
