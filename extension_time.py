import time
import cv2, numpy as np
import imageIO.png
import os.path
import math
from scipy.ndimage import generic_filter


def originalVersion(image_width, image_height, px_array_r, px_array_g, px_array_b):
    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    px_array = applyStdDevFilter(px_array, image_width, image_height)
    return px_array

def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array
def separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height):
    new_array = [[[0 for c in range(3)] for x in range(image_width)] for y in range(image_height)]

    for y in range(image_height):
        for x in range(image_width):
            new_array[y][x][0] = px_array_r[y][x]
            new_array[y][x][1] = px_array_g[y][x]
            new_array[y][x][2] = px_array_b[y][x]
    return new_array
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    # STUDENT CODE HERE

    for h in range(image_height):
        for w in range(image_width):
            greyscale_pixel_array[h][w] = int(
                round(pixel_array_r[h][w] * 0.299 + pixel_array_g[h][w] * 0.587 + pixel_array_b[h][w] * 0.114))

    return greyscale_pixel_array
def applySobelFilter(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)

    Gx = [[1.0, 0.0, -1.0],
          [2.0, 0.0, -2.0],
          [1.0, 0.0, -1.0]]

    Gy = [[1.0, 2.0, 1.0],
          [0.0, 0.0, 0.0],
          [-1.0, -2.0, -1.0]]

    for h in range(1, image_height - 1):
        for w in range(1, image_width - 1):
            gx, gy = 0, 0

            for fh in range(3):
                for fw in range(3):
                    gx += Gx[fh][fw] * pixel_array[h + fh - 1][w + fw - 1]
                    gy += Gy[fh][fw] * pixel_array[h + fh - 1][w + fw - 1]

            result[h][w] = int(abs(gx - gy))

    return result
def applyStdDevFilter(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)

    for h in range(image_height):
        for w in range(image_width):
            r = []

            if h <= 1 or h >= image_height - 2 or w <= 1 or w >= image_width - 2:
                result[h][w] = 0.0
                continue

            for fh in range(5):
                for fw in range(5):
                    r.append(pixel_array[h + fh - 2][w + fw - 2])

            result[h][w] = math.sqrt(sum((i - (sum(r) / len(r))) ** 2 for i in r) / len(r))

    return result

def numpyVersion(image):
    image = grayscaleNormalise(image)
    image = standardDeviationFilter(image)
    return image

def grayscaleNormalise(image):
    result = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    result /= max(result.flatten()) * 255
    return result
def sobelFilter(image):
    Gx = np.array([[1.0, 0.0, -1.0],
                   [2.0, 0.0, -2.0],
                   [1.0, 0.0, -1.0]])

    Gy = np.array([[1.0, 2.0, 1.0],
                   [0.0, 0.0, 0.0],
                   [-1.0, -2.0, -1.0]])

    [rows, columns] = np.shape(image)
    sobel_filtered_image = np.zeros(shape=(rows, columns))

    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, image[i:i + 3, j:j + 3]))
            gy = np.sum(np.multiply(Gy, image[i:i + 3, j:j + 3]))
            sobel_filtered_image[i + 1, j + 1] = np.abs(gx - gy)
    return sobel_filtered_image
def standardDeviationFilter(image):
    image_height, image_width = image.shape[:2]
    pixel_array = image.tolist()
    result = np.zeros((image_height, image_width), dtype=np.float64)

    for h in range(image_height):
        for w in range(image_width):
            r = []

            if h <= 1 or h >= image_height - 2 or w <= 1 or w >= image_width - 2:
                result[h,w] = 0.0
                continue

            for fh in range(5):
                for fw in range(5):
                    r.append(pixel_array[h + fh - 2][w + fw - 2])

            result[h,w] = math.sqrt(sum((i - (sum(r) / len(r))) ** 2 for i in r) / len(r))
    # return 0
    return np.array(result)

def opencvVersion(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = openCVSobelFilter(image)
    image = scipyStdDevFilter(image)
    return image

def openCVSobelFilter(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return image
def scipyStdDevFilter(image):
    return generic_filter(image, np.std, size=5)

def main():
    filename = os.path.abspath(os.getcwd()) + "/images/Barcode2.png"
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)
    start = time.time()
    image = originalVersion(image_width, image_height, px_array_r, px_array_g, px_array_b)
    end = time.time()
    print("Running times of applying Standard Deviation Filter")
    print(f"Time taken (Original Version):\t{end - start:.4f}s")
    image = cv2.imread(filename)
    start2 = time.time()
    image = numpyVersion(image)
    end2 = time.time()
    print(f"Time taken (Numpy Version):\t\t{end2 - start2:.4f}s")
    image = cv2.imread(filename)
    start3 = time.time()
    image = opencvVersion(image)
    end3 = time.time()
    print(f"Time taken (SciPy Version):\t\t{end3 - start3:.4f}s")

main()

