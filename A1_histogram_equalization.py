import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_pdf(im_in):
    # Create normalized intensity histogram from an input image
    no_of_pixels = [0] * 256
    total_pixels = len(im_in)

    for pixel_value in im_in:
        no_of_pixels[pixel_value] += 1

    pdf = []
    for count in no_of_pixels:
        pdf.append(count / total_pixels)

    return pdf


def plot_pdf(pdf):
    plt.plot(range(256), pdf, color='r')
    plt.title('Probability Density Function (PDF) of the image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Probability Density Value')
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
    # Create the cumulative distribution function from an input pdf
    cdf = []
    cdf.append(pdf[0])
    i = 1
    while i < len(pdf):
        cdf.append(cdf[i - 1] + pdf[i])
        i = i + 1

    return cdf


def histogram_equalization(im_in):
    pdf = create_pdf(im_in)  # Your previously implemented function
    cdf = create_cdf(pdf)  # Your previously implemented function
    plot_pdf(pdf)
    plot_cdf(cdf)

    equalized_im = [int(cdf[pixel_value] * 255) for pixel_value in im_in]
    return equalized_im


def plot_histogram_equalized_image(equalized_data):
    equalized_image = np.array(equalized_data).reshape((256, 256)).astype(np.uint8)

    plt.imshow(equalized_image, cmap='grey')
    plt.title('Histogram Equalized Image')
    plt.axis('off')
    plt.show()



img_path = 'indoors.png'
img_data = Image.open(img_path)
img_data = list(img_data.getdata())
equalized_data = histogram_equalization(img_data)
plot_histogram_equalized_image(equalized_data)
pdf_equalized = create_pdf(equalized_data)
plot_pdf(pdf_equalized)
cdf_equalized = create_cdf(pdf_equalized)
plot_cdf(cdf_equalized)

equalized_data_of_equalized_data = histogram_equalization(equalized_data)
plot_histogram_equalized_image(equalized_data_of_equalized_data)
pdf_equalized = create_pdf(equalized_data_of_equalized_data)
plot_pdf(pdf_equalized)
cdf_equalized = create_cdf(pdf_equalized)
plot_cdf(cdf_equalized)

img_path = 'my_image.png'
img_data = Image.open(img_path)
resized_image = img_data.resize((256, 256))
grayscale_image = resized_image.convert('L')
grayscale_image.save('my_image_resized.png')
img_data = list(grayscale_image.getdata())
equalized_data = histogram_equalization(img_data)
plot_histogram_equalized_image(equalized_data)
pdf_equalized = create_pdf(equalized_data)
plot_pdf(pdf_equalized)
cdf_equalized = create_cdf(pdf_equalized)
plot_cdf(cdf_equalized)