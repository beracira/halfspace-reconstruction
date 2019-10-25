from image_gen import Image_Generator
from collections import deque
import numpy as np
from PIL import Image

image_size = 512

ig = Image_Generator(size=image_size, mode='curve', 
    noisy=True, blur=True)
original = ig.get_new_image()

queue = deque()

queue.append((0, 0))


result = np.zeros((image_size, image_size), dtype=np.uint8)
added = np.zeros((image_size, image_size), dtype=np.uint8)

while queue:
  cur = queue.popleft()
  x, y = cur
  if original[x, y] >= 196:
    result[x, y] = 255
  if x + 1 < image_size and added[x + 1, y] == 0:
    queue.append((x + 1, y))
    added[x + 1, y] = 1
  if y + 1 < image_size and added[x, y + 1] == 0:
    queue.append((x, y + 1))
    added[x, y + 1] = 1

result = Image.fromarray(result, 'L')
result.save("processed.png")
original = Image.fromarray(original, 'L')
original.save("original.png")
print (result)
