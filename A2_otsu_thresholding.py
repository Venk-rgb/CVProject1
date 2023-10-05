import math

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def manual_threshold(im_in, threshold):
    # Threshold image with the threshold of your choice
    manual_thresh_img = im_in
    for i in range(len(im_in)):
        manual_thresh_img[i] = 255 if im_in[i] > threshold else 0
    return manual_thresh_img


def create_pdf(im_in, im_size):
    # Create normalized intensity histogram from an input image
    no_of_pixels = [0] * 256
    total_pixels = len(im_in)

    for pixel_value in im_in:
        no_of_pixels[pixel_value] += 1

    pdf = []
    for count in no_of_pixels:
        pdf.append(count / total_pixels)

    return pdf


def plot_pdf(pdf, im_size):
    plt.plot(range(256), pdf, color='r')
    plt.title('Probability Density Function (PDF) of the image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Probability Density Value')
    plt.xlim(0, 255)
    plt.show()


def otsu_threshold(im_in, im_size):
    # Create Otsu thresholded image
    pdf = create_pdf(im_in, im_size)
    max_variance = float('-inf')
    otsu_threshold_value = 0
    inter_class_variances = []

    for threshold_value in range(im_size):
        background_probability = sum(pdf[:threshold_value])
        foreground_probability = sum(pdf[threshold_value:])

        if background_probability == 0 or foreground_probability == 0:
            inter_class_variances.append(0)
            continue

        total_sum = sum(i * p for i, p in enumerate(pdf))

        background_sum = sum(i * p for i, p in enumerate(pdf[:threshold_value]))

        background_mean = background_sum / background_probability
        foreground_mean = (total_sum - background_sum) / foreground_probability

        inter_class_variance = background_probability * foreground_probability * (
                (background_mean - foreground_mean) ** 2)

        inter_class_variances.append(inter_class_variance)

        if inter_class_variance > max_variance:
            max_variance = inter_class_variance
            otsu_threshold_value = threshold_value - 1

    print("Otsu threshold value obtained by my algorithm = ", otsu_threshold_value)
    otsu_threshold_image = im_in
    for i in range(len(im_in)):
        otsu_threshold_image[i] = 255 if im_in[i] > otsu_threshold_value else 0
    return otsu_threshold_image, inter_class_variances


def plot_image(img_data, im_size):
    equalized_image = np.array(img_data).reshape((im_size, im_size)).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=im_size - 1)
    plt.title('Image after thresholding')
    plt.axis('off')
    plt.show()


def plot_inter_class_variance(thresholds, inter_class_variances):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, inter_class_variances)
    plt.title('Inter-Class Variance vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Inter-Class Variance')
    plt.show()


img_path = 'b2_a.png'

img_data = Image.open(img_path)
img_data_for_otsu = img_data

img_data = list(img_data.getdata())
im_size = int(math.sqrt(len(img_data)))

pdf = create_pdf(img_data, im_size)
plot_pdf(pdf, im_size)

choice = int(input('Enter which thresholding you want to apply: (1 for manual, 2 for otsu)'))

if choice == 1:
    threshold_img_manual = manual_threshold(img_data, 110)
    plot_image(threshold_img_manual, im_size)
elif choice == 2:
    threshold_img_otsu, inter_class_variances = otsu_threshold(img_data, im_size)

    plot_image(threshold_img_otsu, im_size)

    thresholds = list(range(im_size))

    otsu_threshold, image_result = cv2.threshold(np.array(img_data_for_otsu), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Otsu threshold value obtained by using CV = ", otsu_threshold)

    plot_inter_class_variance(thresholds, inter_class_variances)
