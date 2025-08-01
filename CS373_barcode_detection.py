# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
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


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


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


def applyStdDevFilter(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)

    for h in range(image_height):
        for w in range(image_width):
            r = []

            if h == 0 or h >= image_height - 1 or w == 0 or w >= image_width - 1:
                result[h][w] = 0.0
                continue

            for fh in range(3):
                for fw in range(3):
                    r.append(pixel_array[h + fh - 1][w + fw - 1])

            result[h][w] = math.sqrt(sum((i - (sum(r) / len(r))) ** 2 for i in r) / len(r))

    return result


def computeBorderBoundaryPadding(pixel_array, image_width, image_height):
    borderboundarypadding = createInitializedGreyscalePixelArray(image_width + 2, image_height + 2)
    for h in range(image_height):
        for w in range(image_width):
            borderboundarypadding[h + 1][w + 1] = pixel_array[h][w]
    for w in range(2):
        borderboundarypadding[w] = borderboundarypadding[1]
        borderboundarypadding[-w - 1] = borderboundarypadding[-2]
    for h in range(len(borderboundarypadding)):
        row = borderboundarypadding[h]
        row[0] = row[1]
        row[-1] = row[-2]
    return borderboundarypadding


def applyGaussianFilter(pixel_array, image_width, image_height):
    padded_array = computeBorderBoundaryPadding(pixel_array, image_width, image_height)
    result = createInitializedGreyscalePixelArray(image_width, image_height)

    gauss = [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]]

    for w in range(image_width):
        for h in range(image_height):
            raw = 0.0
            for fw in range(3):
                for fh in range(3):
                    raw += padded_array[h + fh][w + fw] * gauss[fh][fw]
            result[h][w] = raw / 16

    return result


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for h in range(image_height):
        for w in range(image_width):
            if pixel_array[h][w] >= threshold_value:
                result[h][w] = 255
            else:
                result[h][w] = 0
    return result


def applyErosion(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)

    for h in range(2, image_height - 2):
        for w in range(2, image_width - 2):
            value = 255
            for fh in range(5):
                for fw in range(5):
                    value = min(value, pixel_array[h + fh - 2][w + fw - 2])
            if (value != 0):
                result[h][w] = 255
            else:
                result[h][w] = 0
    return result


def applyDilation(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)

    for h in range(2, image_height - 2):
        for w in range(2, image_width - 2):
            value = 0
            for fh in range(5):
                for fw in range(5):
                    value = max(value, pixel_array[h + fh - 2][w + fw - 2])
            if (value != 0):
                result[h][w] = 255
            else:
                result[h][w] = 0
    return result


def checkSide(w,h, image_width, image_height):
    return 0 <= w < image_width and 0 <= h < image_height


def checkAllSides(w, h, image_width, image_height):
    availableSides = []
    if checkSide(w - 1, h, image_width, image_height):    # left
        availableSides.append((w - 1, h))
    if checkSide(w + 1, h, image_width, image_height):    # right
        availableSides.append((w + 1, h))
    if checkSide(w, h - 1, image_width, image_height):    # top
        availableSides.append((w, h - 1))
    if checkSide(w, h + 1, image_width, image_height):    # bottom
        availableSides.append((w, h + 1))
    return availableSides


def bfsCount(label, w, h, visited, result, pixel_array, image_width, image_height):
    queue = Queue()
    queue.enqueue((w, h))
    visited.add((w, h))

    count = 0

    while not queue.isEmpty():
        currW, currH = queue.dequeue()
        result[currH][currW] = label
        count += 1

        for sideW, sideH in checkAllSides(currW, currH, image_width, image_height):
            if pixel_array[sideH][sideW] >= 1 and (sideW, sideH) not in visited:
                queue.enqueue((sideW, sideH))
                visited.add((sideW, sideH))

    return count


def labelAnalysis(result, pixel_count, image_width, image_height):
    dim_dict = {}
    dimensions = []

    minDensity = 0.6
    minArea = 3000

    # for key in pixel_count:
    for h in range(image_height):
        for w in range(image_width):
            if result[h][w] != 0:
                if result[h][w] in dim_dict:
                    dim_dict[result[h][w]][0] = max(dim_dict[result[h][w]][0], w)
                    dim_dict[result[h][w]][1] = max(dim_dict[result[h][w]][1], h)
                    dim_dict[result[h][w]][2] = min(dim_dict[result[h][w]][2], w)
                    dim_dict[result[h][w]][3] = min(dim_dict[result[h][w]][3], h)
                else:
                    dim_dict[result[h][w]] = [w, h, w, h]

    for label in dim_dict:
        maxX, maxY, minX, minY = dim_dict[label]
        density = pixel_count[label] / (abs(maxX - minX) * abs(maxY - minY))
        ratio = max(abs(maxX - minX), abs(maxY - minY)) / min(abs(maxX - minX), abs(maxY - minY))
        # print(f"label: {label}\tratio: {ratio:.2f}\tdensity: {density:.2f}\tarea: {pixel_count[label]:.2f}\tmaxX: {maxX}\tmaxY: {maxY}\tminX: {minX}\tminY: {minY}")
        if density > minDensity and 0.8 < ratio < 1.8 and pixel_count[label] > minArea:
            dimensions.append((label, maxX, maxY, minX, minY))

    return dimensions


def applyConnectedComponent(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    pixel_count = {}
    visited = set()

    label = 1
    for h in range(image_height):
        for w in range(image_width):
            if pixel_array[h][w] >= 1 and (w, h) not in visited:
                pixel_count[label] = bfsCount(label, w, h, visited, result, pixel_array, image_width, image_height)
                label += 1

    box_dimensions = labelAnalysis(result, pixel_count, image_width, image_height)

    return result, box_dimensions


# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():
    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode7"
    input_filename = "images/" + filename + ".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename + "_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)


    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    # STUDENT IMPLEMENTATION here

    # Step 1. Convert to Greyscale and Normalise
    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    # Step 2. Apply Standard Deviation Filter (Option 2)
    px_array = applyStdDevFilter(px_array, image_width, image_height)
    # Step 3. Apply Gaussian Filter
    px_array = applyGaussianFilter(px_array, image_width, image_height)
    px_array = applyGaussianFilter(px_array, image_width, image_height)
    # Step 4. Apply Thresholding
    px_array = computeThresholdGE(px_array, 20, image_width, image_height)
    # Step 5. Apply Erosion & Dilation
    px_array = applyErosion(px_array, image_width, image_height)
    px_array = applyDilation(px_array, image_width, image_height)
    px_array = applyDilation(px_array, image_width, image_height)
    # Step 6. Apply Connected Component
    px_array, box_dimensions = applyConnectedComponent(px_array, image_width, image_height)

    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height))

    for l in box_dimensions:
        _, bbox_max_x, bbox_max_y, bbox_min_x, bbox_min_y = l

        rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                         edgecolor='g', facecolor='none')
        axs1[1, 1].add_patch(rect)


    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()