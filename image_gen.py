from PIL import Image
import numpy as np
import logging

class Image_Generator(object):
    """docstring for Image_Generator."""

    def __init__(self, size=512, mode=None, noisy=False, blur=False):
        super(Image_Generator, self).__init__()
        self.mode = mode
        self.size = size
        self.noisy = noisy
        self.blur = blur


    # an image is a 2d numpy array
    def get_new_image(self):
        if self.mode is None:
            image = np.random.randint(128, size=(self.size, self.size), dtype=np.uint8)
        elif self.mode == 'halfspace':
            image = np.zeros((self.size, self.size), dtype=np.uint8)

            # TODO: adjust the ratio between different types of images.
            first = np.random.randint(4)
            second = np.random.randint(4)
            
            if first != second:
                coor_1 = np.random.randint(self.size)
                coor_2 = np.random.randint(self.size)
                set_1 = ((coor_1, 0), (0, coor_1), (coor_1, self.size - 1), (self.size - 1, coor_1))
                set_2 = ((coor_2, 0), (0, coor_2), (coor_2, self.size - 1), (self.size - 1, coor_2))

                x_1 = set_1[first][0]
                y_1 = set_1[first][1]
                x_2 = set_2[second][0]
                y_2 = set_2[second][1]

                # TODO: maybe fix the vertical line case?
                if x_1 == x_2: x_1 += 0.01

                for x in range(self.size):
                    for y in range(self.size):

                        if y - y_1 > (y_2 - y_1) / (x_2 - x_1) * (x - x_1):
                            image[x, y] = 255
                        else:
                            image[x, y] = 0 

        if self.noisy is True:
            mask = np.random.randint(64, size=(self.size, self.size), dtype=np.uint8)

            for x in range(self.size):
                for y in range(self.size):
                    if image[x, y] == 255:
                        image[x, y] -= mask[x, y]
                    else:
                        image[x, y] += mask[x, y]

        if self.blur is True:
            temp = np.zeros((self.size, self.size), dtype=np.uint8)

            blur_radius = 10
            for x in range(self.size):
                for y in range(self.size):
                    acc = 0
                    t, b = max(0, x - blur_radius), min(self.size - 1, x + blur_radius)
                    l, r = max(0, y - blur_radius), min(self.size - 1, y + blur_radius)
                    sub_arr = image[t:b, l:r]
                    temp[x, y] = np.mean(sub_arr)

            image = temp

        return image


if __name__ == '__main__':
    ig = Image_Generator(mode='halfspace', noisy=True, blur=True)
    img = ig.get_new_image()
    img = Image.fromarray(img, 'L')
    img.save("temp.png")
    print (img)
