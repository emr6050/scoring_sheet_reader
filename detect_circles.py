import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import cv2
from pdf2image import convert_from_path
import math

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


def detect_and_draw_circles(img_file, wThresh, hThresh):
    file_img = cv2.imread(img_file)

    # circle constraints
    minR = 12
    maxR = 18
    minDist = 35
    param1 = 50
    param2 = 14

    # restrict the contour detection to a scoring_region
    (rows, columns) = file_img.shape[:2]
    widthThreshold = int(columns*wThresh)
    heightThreshold = int(rows*hThresh)
    region = file_img[heightThreshold:rows, widthThreshold:columns]

    # process the image
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    bwImg = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow('black & white image', cv2.resize(bwImg, None, fx=.5, fy=.5))
    # cv2.imwrite('bwImg_region.png', bwImg)
    # cv2.waitKey(0)

    # Hough Transform to detect circles
    circles = cv2.HoughCircles(bwImg, cv2.HOUGH_GRADIENT,
                               dp=1, minDist=minDist,
                               param1=param1, param2=param2,
                               minRadius=minR, maxRadius=maxR)
    if circles is None:
        print("Error: no circles correspond to the input parameters!")
        return
    circles = np.int16(np.around(circles))
    print("number of detected circles:", len(circles[0, :]))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(region, (i[0], i[1]), i[2], BLUE, 2)
        # draw the center of the circle
        # cv2.circle(region, (i[0], i[1]), 2, RED, 3)

    cv2.imshow('detected circles', cv2.resize(region, None, fx=.5, fy=.5))
    # cv2.imwrite('images/pg2-filled-detected.png', region)
    cv2.waitKey(0)

    # # TODO: check circles groupings
    # # sort by proximity
    # circles = circles[0, :]
    # circles2 = sorted(circles, key=lambda c: math.sqrt(
    #     (c[0] - c[0])**2 + (c[1] - c[1])**2))
    # for i in range(len(circles2)-1):
    #     distance = math.sqrt(
    #         (circles2[i+1][0] - circles2[i][0])**2 + (circles2[i+1][1] - circles2[i][1])**2)
    #     print("distance between consecutive points:", distance)


# pdf_file = "special/assessment-filled.pdf"
# pages = convert_from_path(pdf_file)
# count = 1
# for page in pages:
#     page_file = "special/temp/page_"+str(count)+"-filled.png"
#     page.save(page_file, "PNG")
#     count = count+1

# image details
page_file = 'special/temp/page_1-filled.png'  # 'images/pg1-blank.PNG'
wThresh = 0.0  # 0.7
hThresh = 0.0
detect_and_draw_circles(page_file, wThresh=wThresh, hThresh=hThresh)

# # find contours
# cnts, hierarchy = cv2.findContours(bwImg, cv2.RETR_EXTERNAL,
#                                    cv2.CHAIN_APPROX_SIMPLE)
# print("Number of detected contours:", len(cnts))
