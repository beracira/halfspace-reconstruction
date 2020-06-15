import cv2 as cv
from image_gen import Image_Generator
from collections import deque
import numpy as np
from PIL import Image
from timeit import default_timer as timer


def similar_color(x, y):
    return(np.uint8(np.absolute(np.int16(x)-np.int16(y)))) > 32
    # return np.abs(x - y) > 32

def find_change_point(line):
    counter = 0
    i = 0
    j = line.shape[0] - 1
    while counter < 20 and i < j:
        # global pixel_counter
        # pixel_counter += 2
        mid = (i + j) // 2
        if not similar_color(line[mid], line[mid + 1]):
            return mid
        elif similar_color(line[i], line[mid]):
            i = mid + 1
        else:
            j = mid
    return (i + j) // 2

def is_on_line(p1, p2, target):
    if all(p1 == p2):
        if all(target == p1): 
            return 1
        return 0
    d = np.linalg.norm(np.cross(p2 - p1, p1 - target)) / np.linalg.norm(p2 - p1)
    if d < 1:
        return 1
    return 0

counter = 0
def get_black_pts(img, x, y, counter_arr):
    global counter
    counter += 1
    height, width = img.shape
    top_left = img[0, 0]
    top_right = img[0, -1]
    bot_left = img[-1, 0]
    bot_right = img[-1, -1]
    counter_arr[x, y] = 1
    counter_arr[x + height - 1, y] =1
    counter_arr[x, y + width - 1] = 1
    counter_arr[x + height - 1, y + width - 1] = 1  

    corner_arr = np.array((top_left, top_right, bot_left, bot_right))
    avg = np.mean(corner_arr)
    return *(corner_arr > avg * 1.1), corner_arr[corner_arr > avg * 1.1].shape[0]


def recursion_helper(retval, img, counter_arr, x, y, min_length):
    height, width = img.shape
    retval[0:height // 2, 0:width // 2] = find_halfspace(img[0:height // 2, 0:width // 2], counter_arr, x, y, min_length)
    retval[height // 2:, 0:width // 2] = find_halfspace(img[height // 2:, 0:width // 2], counter_arr, x + height // 2, y, min_length)
    retval[0:height // 2, width // 2:] = find_halfspace(img[0:height // 2, width // 2:], counter_arr, x, y + width // 2, min_length)
    retval[height // 2:, width // 2:] = find_halfspace(img[height // 2:, width // 2:], counter_arr, x + height // 2, y + width // 2, min_length)

def recursion_wrapper(args):
    return (find_halfspace(*args), args[1])

def find_halfspace(img, counter_arr, x=0, y=0, min_length=4):
    """
    returns an image with black background and boundary marked in white dots
    """

    height, width = img.shape
    retval = np.zeros((height, width), dtype=np.uint8)
    if height >= min_length and width >= min_length:
        recursion_helper(retval, img, counter_arr, x, y, min_length)
        return retval

    if height <= 1: 
        return retval

    top_left, top_right, bot_left, bot_right, num_black_pts = get_black_pts(img, x, y, counter_arr)

    if num_black_pts == 0 or num_black_pts == 4:
        # recursion_helper(retval, img, counter_arr, x, y, min_length)
        pass
    elif num_black_pts == 1 or num_black_pts == 3:
        # find boundary
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
            recursion_helper(retval, img, counter_arr, x, y, min_length)
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

from multiprocessing import Pool
import sys

input_filename = 'result/' + sys.argv[1] 
if __name__ == '__main__':
    with Pool(8) as p:


        print ("Generating testing image pair...")
        image = cv.imread(input_filename, cv.IMREAD_GRAYSCALE)
        height, width = image.shape

        blur = False # high pass filter to improve accuracy
        if blur:
            temp = np.zeros(image.shape, dtype=np.uint8)
            blur_radius = 1
            for x in range(height):
                for y in range(width):
                    acc = 0
                    t, b = max(0, x - blur_radius), min(height - 1, x + blur_radius)
                    l, r = max(0, y - blur_radius), min(width - 1, y + blur_radius)
                    sub_arr = image[t:b, l:r]
                    temp[x, y] = np.mean(sub_arr)
            image = temp

        print ("Finding boundary on testing image...")
        counter_arr = np.zeros(image.shape, dtype=np.uint8)

        data = []
        for i in range(5):
            data += [(image.copy(), counter_arr.copy(), 0, 0, 2 ** (2 + i))]

        ans = p.map(recursion_wrapper, data)
        for i, each in enumerate(ans):
            edges = each[0]
            counter_arr = each[1]
            edges *= 255
            img = Image.fromarray(edges, 'L')
            output_filename = 'result/edges_' + repr(data[i][-1]) + '_' + sys.argv[1]
            print ("Total number of pixels in the image:", height * width)
            print ("Total number of pixels seen:", np.sum(counter_arr))
            print ("Total %% of pixels seen:", np.sum(counter_arr) / height / width * 100)
            print ('saving to', output_filename)
            print ('')
            img.save(output_filename)


