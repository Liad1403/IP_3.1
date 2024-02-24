import cv2
import numpy as np
import os
import ssim
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim

def calculate_mse(image1, image2):
    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Compute MSE
    mse = np.sum((image1 - image2)**2) / float(image1.size)
    print("The MSE is: ",mse)

def apply_mean_filter(image, kernel_size):

    # Apply mean filter
    smoothed_image = cv2.blur(image, kernel_size)
    cv2.imshow('Mean Filter', smoothed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return smoothed_image

def apply_sharp(image):
    # Define a sharpening kernel
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]], dtype=np.float32)

    # Apply convolution using cv2.filter2D
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

    cv2.imshow('Sharpened Image', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return sharpened_image

def apply_cyclic_pattern(image):

    filtered = np.roll(image, image.shape[0] // 2, axis=0)

    cv2.imshow('Upward Padding Result', filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # Get image dimensions
    rows, cols = image.shape

    # Define the shape of the square matrix
    matrix_shape = (501, 501)

    # Create a square matrix filled with zeros
    convolution_mask = np.zeros(matrix_shape)
    convolution_mask[250, 250] = 1

    # Normalize the mask to ensure proper scaling
    convolution_mask /= np.sum(convolution_mask)

    # Perform convolution using scipy.signal.convolve2d with 'full' mode
    convolution_result_full = convolve2d(image, convolution_mask, mode='full', boundary='fill')

    # Extract the central part with the same size as the input image
    convolution_result = convolution_result_full[:rows, :cols]

    # Convert the result back to uint8 format
    convolution_result = np.uint8(np.absolute(convolution_result))
'''
    return filtered


def apply_convolution_with_Gaussian_Mask(image):

    # Define a custom Laplacian kernel (example: 3x3 kernel)
    custom_laplacian_kernel = np.array([[-1/16, -1/16, -1/16, -1/16, -1/16],
                                  [-1/16,  1/8,   1/8,   1/8,  -1/16],
                                  [-1/16,  1/8,   1,     1/8,  -1/16],
                                  [-1/16,  1/8,   1/8,   1/8,  -1/16],
                                  [-1/16, -1/16, -1/16, -1/16, -1/16]], dtype=np.float32)

    # Normalize the kernel to ensure proper scaling
    # custom_laplacian_kernel /= np.sum(np.abs(custom_laplacian_kernel))

    # Apply the custom Laplacian filter
    afterConvImage = cv2.filter2D(image, cv2.CV_64F, custom_laplacian_kernel)
    laplacian_image = cv2.normalize(image-afterConvImage, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the result back to uint8 format
    laplacian_image = np.uint8(np.absolute(laplacian_image))
    cv2.imshow('apply_convolution_with_Gaussian_Mask', laplacian_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return laplacian_image

def apply_custom_laplacian(image):
    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    # Apply convolution using OpenCV's filter2D function
    filtered_image = cv2.filter2D(image, -1, laplacian_kernel)
    laplacian_image = np.uint8(np.absolute(filtered_image))

    cv2.imshow('Laplacian Filtered Image', laplacian_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filtered_image

def apply_gaussian_blur(image, kernel_size, sigma):

    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)

    return blurred_image

def median_filter(image, window_size):
    
    result = cv2.medianBlur(image, window_size)
    cv2.imshow('gaussian Mask Convolution', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

def gaussianMaskConvolution(image):
    # Define the size and standard deviation of the Gaussian kernel
    kernel_size = 5
    sigma = 1.0

    # Create a Gaussian kernel using cv2.getGaussianKernel
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)

    # Apply convolution using cv2.filter2D with the Gaussian kernel
    convolution_result = cv2.filter2D(image, -1, gaussian_kernel * gaussian_kernel.T)
    cv2.imshow('gaussianMaskConvolution', convolution_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def apply_delta_mask(image):
    laplacian_kernel = np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]], dtype=np.float32)

    # Apply convolution using cv2.filter2D with the Laplacian kernel
    filtered_image = cv2.filter2D(image, cv2.CV_64F, laplacian_kernel)

    # Normalize the result to be in the range [0, 255]
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the result to uint8 format
    filtered_image = np.uint8(filtered_image)

    cv2.imshow('Delta Mask Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return filtered_image

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    realImagePath = os.path.join('images', '1.jpg')
    realImage = cv2.imread(realImagePath,cv2.IMREAD_GRAYSCALE)

    image_1 = cv2.imread(os.path.join('images', 'image_1.jpg'), cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(os.path.join('images', 'image_2.jpg'), cv2.IMREAD_GRAYSCALE)
    image_3 = cv2.imread(os.path.join('images', 'image_3.jpg'), cv2.IMREAD_GRAYSCALE)
    image_4 = cv2.imread(os.path.join('images', 'image_4.jpg'), cv2.IMREAD_GRAYSCALE)
    image_5 = cv2.imread(os.path.join('images', 'image_5.jpg'), cv2.IMREAD_GRAYSCALE)
    image_6 = cv2.imread(os.path.join('images', 'image_6.jpg'), cv2.IMREAD_GRAYSCALE)
    image_7 = cv2.imread(os.path.join('images', 'image_7.jpg'), cv2.IMREAD_GRAYSCALE)
    image_8 = cv2.imread(os.path.join('images', 'image_8.jpg'), cv2.IMREAD_GRAYSCALE)
    image_9 = cv2.imread(os.path.join('images', 'image_9.jpg'), cv2.IMREAD_GRAYSCALE)
    '''
    # compare to image_1
    verticalMaskConv = apply_mean_filter(realImage, (1180, 2))
    calculate_mse(image_1,verticalMaskConv)
    # compare to image_2

    meanFilteredImage = apply_mean_filter(realImage,(8,8))
    calculate_mse(image_2, meanFilteredImage)

    # compare to image_3
    medianFilterImage = median_filter(realImage, 9)
    calculate_mse(image_3, medianFilterImage)

    # compare to image_4
    verticalMaskConv = apply_mean_filter(realImage, (2,13))
    calculate_mse(image_4, verticalMaskConv)
    # compare to image_5
    gausMaskFilter = apply_convolution_with_Gaussian_Mask(realImage)
    calculate_mse(image_5, gausMaskFilter)
    '''
    # compare to image_6
    laplacianFilterImage = apply_custom_laplacian(realImage)
    calculate_mse(image_6, laplacianFilterImage)

    # compare to image_7
    cyclicImage = apply_cyclic_pattern(realImage)
    calculate_mse(image_7, cyclicImage)
    # compare to image_8
    deltaMaskFilterImage = apply_delta_mask(realImage)
    calculate_mse(image_8, deltaMaskFilterImage)
    # compare to image_9
    sharpenImage = apply_sharp(realImage)
    calculate_mse(image_9, sharpenImage)
    # gaussianMaskConvolutionImage = gaussianMaskConvolution(realImage)
    # laplacianFilterImage = apply_custom_laplacian(realImage)

