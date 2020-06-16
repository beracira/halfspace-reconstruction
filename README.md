# halfspace-reconstruction
A computer vision project that aims to solve half space reconstruction of grayscale images. 

# Usage
## If you have docker installed...

[run.sh](run.sh) builds the image and mounts src direction into the container. Move the image you want to use into ``src/result`` on the local machine and run within ``src`` 

```
python3 binary_search.py imagename
```

The resulting image will be in ``src/result``.

## If you do not have docker...
[requirements.txt](requirements.txt) is provided for ``pip``. After install all the necessary packages, proceed on the local machine with instructions in previous section.

# Note
The script now support only ``.png`` or ``.jpg`` with grayscale image.
