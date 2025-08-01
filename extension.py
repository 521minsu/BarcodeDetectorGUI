### ---------------------------------------------------------------------------###
# extension.py                                                                   #
# Author: Minsu Kim / mkim844 / 447391679                                        #
# This file creates a GUI for the user to select the image and apply various     #
# image processing techniques for a barcode detection.                           #
#                                                                                #
# Also available in Windows Executable format in the folder for easier use.      #
#                                                                                #
# Although I have recreated all the pipelines for the image processing in        #
# OpenCV and other external libraries, it is not guaranteed that this program    #
# will run smoothly on all computers due to resource limitations.                #
#                                                                                #
# Required Libraries:                                                            #
#  - PySimpleGUI                                                                 #
#  - OpenCV                                                                      #
#  - Numpy                                                                       #
#  - pyzbar                                                                      #
### ---------------------------------------------------------------------------###
import PySimpleGUI as sg

import os.path
import math

import cv2
import numpy as np

enablepyzbar = True

if enablepyzbar:
    from pyzbar import pyzbar

def grayscaleNormalise(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    return result
def sobelFilter(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    result = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return result
def standardDeviationFilter(image):     # SUPER SLOW
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
    return np.array(result)

def connectedComponentAnalysis(frame, minDensity, minArea, minRatio, maxRatio):
    (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(frame, 2, cv2.CV_32S)
    result = np.zeros(frame.shape, dtype="uint8")
    rectangles = []

    # Loop through each component
    for i in range(1, totalLabels):

        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]
        width, height = values[i, cv2.CC_STAT_WIDTH], values[i, cv2.CC_STAT_HEIGHT]
        left, top = values[i, cv2.CC_STAT_LEFT], values[i, cv2.CC_STAT_TOP]
        ratio, density = max(width, height) / min(width, height), (area / (width * height)) * 100

        if area > minArea and density > minDensity and minRatio < ratio < maxRatio:
            componentMask = (label_ids == i).astype("uint8") * 255
            rectangles.append(((left, top), (left + width, top + height)))
            result = cv2.bitwise_or(result, componentMask)

    return result, rectangles

def drawRectangle(image, rectangles, colour):
    for rectangle in rectangles:
        image = cv2.rectangle(image, rectangle[0], rectangle[1], colour, 4)
    return image

def cropImage(image, rectangles):
    result = []
    for rectangle in rectangles:
        result.append(image[rectangle[0][1]:rectangle[1][1], rectangle[0][0]:rectangle[1][0]])
    return result

def readBarcode(window, image, rectangles=None):
    if enablepyzbar:
        if rectangles:
            images = cropImage(image, rectangles)
            barcodes = []
            for img in images:
                barcode = pyzbar.decode(img)
                if barcode:
                    for b in barcode:
                        barcodes.append(b.data.decode("utf-8"))
            window["-READ BARCODE-"].update('\n'.join(barcodes))
        else:
            barcodes = []
            barcode = pyzbar.decode(image)
            if barcode:
                for b in barcode:
                    barcodes.append(b.data.decode("utf-8"))
            window["-FULL BARCODE-"].update('\n'.join(barcodes))

def main():
    sg.theme("LightGreen")

    # Define the window layout
    file_list_column = [
        [
            sg.Text("Image Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
            )
        ],
        [sg.HSeparator(), ],
        [sg.Text("EXPERIMENTAL FEATURE (Reading Barcode)"), ],
        [sg.HSeparator(), ],
        [sg.Text("Potential Barcodes (Read using entire image):"),],
        [sg.Text("None", size=(40, 10), key="-FULL BARCODE-")],
        [sg.HSeparator(), ],
        [sg.Text("Detected Barcodes (Only updates with box drawn):"),],
        [sg.Text("None", size=(40, 10), key="-READ BARCODE-")],

    ]

    image_modifier_column  = [
        [sg.Text("Step 1. Convert to GreyScale and Normalise:"),],
        [
            sg.Checkbox('To Greyscale', size=(50, 1), key="-GREYSCALE-"),
        ],
        [sg.HSeparator(),],
        [
            sg.Text("Step 2.Edge Detection (Std Dev won't apply on a colour image!)"),
            sg.Checkbox('TURN ON', size=(10, 1), key="-EDGE-"),
        ],
        [
            sg.Radio('3x3 Sobel Filter', "Radio",size=(15, 1), key="-SOBEL-"),
            sg.Radio('5x5 Standard Deviation Filter (NOT RECOMMENDED!!)', "Radio",size=(40, 1), key="-STDDEV-"),
        ],
        [sg.HSeparator(),],
        [sg.Text("Step 3. Guassian Filter:"), sg.Checkbox('Apply Blur Filter', size=(15, 1), key="-BLUR-"),],
        [
            sg.Text('Filter Size', size=(18, 1)),
            sg.Slider(
                (3, 31),
                3,
                2,
                orientation="h",
                size=(35, 15),
                key="-BLUR kSIZE SLIDER-",
            ),
        ],
        [
            sg.Text('Sigma', size=(18, 1)),
            sg.Slider(
                (1, 24),
                1,
                1,
                orientation="h",
                size=(35, 15),
                key="-BLUR SIGMA SLIDER-",
            ),
        ],
        [sg.HSeparator(),],
        [sg.Text("Step 4. Threshold the image:"),],
        [
            sg.Checkbox('Threshold', size=(15, 1), key="-THRESH-"),
            sg.Slider(
                (0, 255),
                25,
                1,
                orientation="h",
                size=(35, 15),
                key="-THRESH SLIDER-",
            ),
        ],
        [sg.HSeparator(),],
        [sg.Text("Step 5. Erosion and Dilation:"), ],
        [
            sg.Checkbox('No. of Erosion', size=(15, 1), key="-EROSION-"),
            sg.Slider(
                (1,10),
                1,
                1,
                orientation="h",
                size=(35, 15),
                key="-EROSION SLIDER-",
            ),
        ],
        [
            sg.Checkbox('No. of Dilation', size=(15, 1), key="-DILATION-"),
            sg.Slider(
                (1, 10),
                1,
                1,
                orientation="h",
                size=(35, 15),
                key="-DILATION SLIDER-",
            ),
        ],
        [sg.HSeparator(),],
        [sg.Text("Step 6. Connected Component Analysis:"), sg.Checkbox('Show Label', size=(15, 1), key="-LABEL-"), ],
        [
            sg.Text('Min Density', size=(18, 1)),
            sg.Slider(
                (1, 100),
                60,
                1,
                orientation="h",
                size=(35, 15),
                key="-CCA MIN DENSITY-",
            ),
        ],
        [
            sg.Text('Min Area', size=(18, 1)),
            sg.Slider(
                (0, 20000),
                2500,
                500,
                orientation="h",
                size=(35, 15),
                key="-CCA MIN AREA-",
            ),
        ],
        [
            sg.Text('Min Ratio', size=(18, 1)),
            sg.Slider(
                (0, 5),
                0.8,
                0.1,
                orientation="h",
                size=(35, 15),
                key="-CCA MIN RATIO-",
            ),
        ],
        [
            sg.Text('Max Ratio', size=(18, 1)),
            sg.Slider(
                (0, 5),
                1.8,
                0.1,
                orientation="h",
                size=(35, 15),
                key="-CCA MAX RATIO-",
            ),
        ],
        [sg.HSeparator(),],
        [sg.Text("Step 7. Draw a bounding box:"), ],
        [
            sg.Checkbox('Draw it!', size=(20, 1), key="-DRAW BOX-"),
        ],
        [sg.HSeparator(),],
        [sg.Text("Extras:"), ],
        [
            sg.Checkbox('Show Original Image!', size=(30, 1), key="-ORIGINAL-"),
            sg.Checkbox('Trigger Live Cam!', size=(30, 1), key="-LIVE-"),
        ],
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [sg.Text("Your image:")],
        [sg.Image(key="-IMAGE-")],
    ]

    # ----- Full layout -----
    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_modifier_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    window = sg.Window("Barcode Detector Customiser", layout, icon="Icon.ico")


    # Basic HouseKeeping
    filename = os.path.abspath(os.getcwd()) + "/images/Barcode1.png"
    fiveByFiveKernel = np.ones((5, 5), np.uint8)
    cap = cv2.VideoCapture(0)


    # Main Loop
    while True:
        rectangles = None

        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
            # Folder name was filled in, make a list of files in the folder
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                   and f.lower().endswith((".png", ".jpg"))
            ]
            window["-FILE LIST-"].update(fnames)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )

            except:
                pass
        ret, frame = cap.read()
        if not values["-LIVE-"] or not ret:
            frame = cv2.imread(filename)
        readBarcode(window, grayscaleNormalise(frame))
        originalFrame = frame.copy()

        try:

            # Step 1. Convert to GreyScale and Normalise:
            if values["-GREYSCALE-"]:
                frame = grayscaleNormalise(frame)

            # Step 2. Edge Detection:
            if values["-EDGE-"]:
                if values["-SOBEL-"]:
                    frame = sobelFilter(frame)
                    # frame = cv2.filter2D(frame, -1, sobelKernel)
                elif values["-STDDEV-"] and values["-GREYSCALE-"]:
                    frame = standardDeviationFilter(frame)

            # Step 3. Gaussian Filter:
            if values["-BLUR-"]:
                frame = cv2.GaussianBlur(frame, (int(values["-BLUR kSIZE SLIDER-"]), int(values["-BLUR kSIZE SLIDER-"])), values["-BLUR SIGMA SLIDER-"])

            # Step 4. Threshold the image:
            if values["-THRESH-"]:
                frame = cv2.threshold( frame, values["-THRESH SLIDER-"], 255, cv2.THRESH_BINARY )[1]

            # Step 5. Erosion and Dilation:
            if values["-EROSION-"]:
                frame = cv2.erode(frame, fiveByFiveKernel, iterations=int(values["-EROSION SLIDER-"]))
            if values["-DILATION-"]:
                frame = cv2.dilate(frame, fiveByFiveKernel, iterations=int(values["-DILATION SLIDER-"]))

            # Step 6. Connected Component Analysis:
            if values["-LABEL-"]:
                frame, rectangles = connectedComponentAnalysis(frame, values["-CCA MIN DENSITY-"], values["-CCA MIN AREA-"], values["-CCA MIN RATIO-"], values["-CCA MAX RATIO-"])

            # Step 7. Draw a bounding box:
            if ((values["-ORIGINAL-"] and not values["-LIVE-"]) or (values["-ORIGINAL-"] and values["-LIVE-"] and not ret)):
                frame = cv2.imread(filename)

            if values["-DRAW BOX-"] and rectangles is not None:
                colour = (255, 0, 0)
                if values["-ORIGINAL-"]:
                    colour = (0, 255, 0)
                frame = drawRectangle(frame, rectangles, colour)
                readBarcode(window, grayscaleNormalise(originalFrame), rectangles)


        except:
            pass

        # print(frame[0],end="\n\n\n\n")

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)


    window.close()

main()