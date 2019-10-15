# Optical Mark Recognition (OMR) for reading the assessment scoring forms
# created by: Peter Dobbs
# based on: https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/
# TODO: programmatically find scoring region (currently there's a preset threshold)
# TODO: loop over all pages of a pdf document (currently the program has to be adjusted for number of images)
# TODO: check that question without an answer does not mess up the program
# TODO: check if contour has more pixels than should be possible

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# constants
RED = (0, 0, 255)
BLUE = (255, 0, 0)
SCORE_OPTIONS = [0,5,10,15,20,25,30,35,40,45,50,55,60]
RESPONSE_OPTIONS = [True,False]

# variables
subtotal = 0
questionOffset = 0  # useful for multi-page questionaire
answers = {}


def score_assessment_form(file_string, wThresh, hThresh):
    global questionOffset, subtotal, answers
    file_img = cv2.imread(file_string)

    # restrict the contour detection to a scoring_region
    (rows, columns) = file_img.shape[:2]
    widthThreshold = int(columns*wThresh)
    heightThreshold = int(rows*hThresh)
    region = file_img[heightThreshold:rows, widthThreshold:columns]

    # scale, gray, and threshold the image
    scoring_region = cv2.resize(region, None, fx=3, fy=3)
    gray = cv2.cvtColor(scoring_region, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if w >= 20 and h >= 20 and aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
            questionCnts.append(c)

    questionCnts = contours.sort_contours(
        questionCnts, method="top-to-bottom")[0]

    for (q, i) in enumerate(np.arange(0, len(questionCnts), SCORE_OPTIONS.__len__)):
        cnts = contours.sort_contours(
            questionCnts[i:i + SCORE_OPTIONS.__len__])[0]
        bubbled = (0, 0)

        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled == (0, 0) or total > bubbled[0]:
                bubbled = (total, j)

        answers[questionOffset+q+1] = (bubbled[1]+1)
        cv2.drawContours(scoring_region, [cnts[bubbled[1]]], -1, RED, 3)
        subtotal += (bubbled[1]+1)

    questionOffset = q+1

    # draw thresholds for scoring region on the full page
    cv2.line(file_img, (0, heightThreshold),
             (columns, heightThreshold), BLUE, 2)
    cv2.line(file_img, (widthThreshold, 0),
             (widthThreshold, rows), BLUE, 2)
    # show original image and scoring chart to check for errors
    cv2.imshow("Drawing Region on Original Page", cv2.resize(
        file_img, None, fx=0.75, fy=0.75))
    cv2.imshow("Scoring region", cv2.resize(
        scoring_region, None, fx=0.333, fy=0.333))
    cv2.waitKey(1)


score_assessment_form('special/pg2_filled.PNG', 0.7, 0.25)
score_assessment_form('special/pg3_filled.PNG', 0.7, 0.0)

print(subtotal)

print("--finished--")
