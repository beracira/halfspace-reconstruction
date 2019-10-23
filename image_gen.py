from PIL import Image
import numpy as np

class Image_Generator(object):
    """docstring for Image_Generator."""

    def __init__(self, size=512, mode=None):
        super(Image_Generator, self).__init__()
        self.mode = mode
        self.size = size


    # an image is a 2d numpy array
    def get_new_image(self):
        if self.mode is None:
            image = np.random.randint(256, size=(self.size, self.size), dtype=np.uint8)

        return image


if __name__ == '__main__':
    ig = Image_Generator()
    img = ig.get_new_image()
    img = Image.fromarray(img, 'L')
    img.save("temp.png")
    print (img)
