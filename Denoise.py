#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=2PrzKWkqOtU


# Filters work by convolution with a moving window called a kernel.
# Convolution is nothing but multiplication of two arrays of different sizes.
# The image will be of one size and the kernel with be of a different size,
# #usually much smaller than image
# The input pixel is at the centre of the kernel.
# The convolution is performed by sliding the kernel over the image,
# $usually from top left of image.
# Linear filters and non-linear filters.
# Gaussian is an example of linear filter.
# Non-linear filters preserve edges.
# Median filter is an example of non-linear filter.
# The algorithm selects the median value of all the pixels in the selected window
# NLM: https://scikit-image.org/docs/dev/auto_examples/filters/plot_nonlocal_means.html


############################ Denoising filters ###############
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float
from scipy import ndimage as nd
from matplotlib import pyplot as plt
from skimage import io


import numpy as np
import cv2

def Denoise(img):
    img = img_as_float(io.imread(img))
    # Need to convert to float as we will be doing math on the array

    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img_normalized * 255).astype(np.uint8)

    ##### NLM#####

    sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))

    # patch_kw = dict(patch_size=5,
    #                 patch_distance=3,
    #                 multichannel=True)

    denoise_img = denoise_nl_means(img, h=1.15* sigma_est, fast_mode=True,
                                   patch_size=5, patch_distance=3, channel_axis=-1)
    """
    denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                               **patch_kw)
    """
    denoise_img_as_8byte = img_as_ubyte(denoise_img)

    # plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')
    plt.imsave("images/NLM.jpg", denoise_img)

    return denoise_img