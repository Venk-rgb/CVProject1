from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def create_histogram(im_in):
    histogram = np.zeros(256, dtype=int)

    for pixel_value in im_in.ravel():
        histogram[pixel_value] += 1

    pdf = []
    for count in histogram:
        pdf.append(count / sum(histogram))

    return pdf


def plot_histogram(pdf):
    plt.plot(range(256), pdf, color='r')
    plt.title('Histogram of the image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel count')
    plt.xlim(0, 255)
    plt.show()


def plot_cdf(cdf):
    plt.plot(range(256), cdf, color='r')
    plt.title('Cummulative Density Function (CDF) of the image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Cummulative Density Value')
    plt.xlim(0, 255)
    plt.show()


def create_cdf(pdf):
    cdf = np.zeros(256, dtype=float)
    cdf[0] = pdf[0]

    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    return cdf

def histogram_equalization(im_in):
    histogram = create_histogram(im_in)
    cdf = create_cdf(histogram)

    plot_histogram(histogram)
    plot_cdf(cdf)

    equalized_channel = (cdf[im_in] * 255).astype(np.uint8)

    return equalized_channel


image_path = 'color_image2.png'
color_image = Image.open(image_path)

color_array = np.array(color_image)

red_channel = color_array[:, :, 0]
green_channel = color_array[:, :, 1]
blue_channel = color_array[:, :, 2]

equalized_red = histogram_equalization(red_channel)
equalized_green = histogram_equalization(green_channel)
equalized_blue = histogram_equalization(blue_channel)

equalized_color_image = np.dstack((equalized_red, equalized_green, equalized_blue))

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(color_image)
plt.title('Original Color Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(equalized_color_image)
plt.title('Equalized Color Image')
plt.axis('off')

plt.show()
