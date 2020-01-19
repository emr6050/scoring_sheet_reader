# Optical Mark Recognition (OMR) for reading the assessment scoring forms
# created by: Peter Dobbs
# based on: https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/
# TODO: check that question without an answer does not mess up the program
# TODO: check if contour has more pixels than should be possible
#           <- if the detected object is just too big to be one of the bubble answers

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from pdf2image import convert_from_path

# constants
RED = (0, 0, 255)
BLUE = (255, 0, 0)
NUM_ANSWER_OPTIONS = 4

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
    bwImage = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imshow('black and white image', cv2.resize(
        bwImage, None, fx=.15, fy=.15))
    cv2.waitKey(0)

    # detect scoring boxes (bubbles in this case)
    # find contours
    # -- this isn't exclusive enough -- It's picking up letters like 'M' and 'O'
    cnts, hierarchy = cv2.findContours(bwImage, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    print("Number of detected contours:", len(cnts))

    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if w >= 50 and h >= 50 and aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
            questionCnts.append(c)
    questionCnts = contours.sort_contours(
        questionCnts, method="top-to-bottom")[0]
    print("Number of question-related contours:", len(questionCnts))

    cv2.drawContours(scoring_region, questionCnts, -1, BLUE, 6)
    cv2.imshow('black and white with overlayed contours', cv2.resize(
        scoring_region, None, fx=.15, fy=.15))
    # cv2.imwrite('detected_contours.png', scoring_region)
    cv2.waitKey(0)

    for (q, i) in enumerate(np.arange(0, len(questionCnts), NUM_ANSWER_OPTIONS)):
        cnts = contours.sort_contours(
            questionCnts[i:i + NUM_ANSWER_OPTIONS])[0]
        bubbled = (0, 0)

        for (j, c) in enumerate(cnts):
            mask = np.zeros(bwImage.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            mask = cv2.bitwise_and(bwImage, bwImage, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled == (0, 0) or total > bubbled[0]:
                bubbled = (total, j)

        answers[questionOffset+q+1] = (bubbled[1]+1)
        cv2.drawContours(scoring_region, [cnts[bubbled[1]]], -1, RED, 5)
        subtotal += (bubbled[1]+1)

    questionOffset = q+1

    cv2.imshow("Scoring region", cv2.resize(
        scoring_region, None, fx=0.15, fy=0.15))
    # cv2.imwrite('detected_contours_scored.png', scoring_region)
    cv2.waitKey(0)


print("--start--")

# pdf_file = "special/SRS-P.pdf"
# pages = convert_from_path(pdf_file)
# count = 0

# for page in pages:
#     page_file = "special/temp/page_"+str(count)+".png"
#     page.save(page_file, "PNG")
#     count = count+1

# for page_num in range(count):
#     page_file = "special/temp/page_"+str(page_num)+".png"
#     score_assessment_form(page_file, wThresh=0.7, hThresh=0.0)

score_assessment_form("special/temp/page_1.png", wThresh=0.7, hThresh=0.0)

print("Raw Score:", subtotal)

# creating FHIR resource for patient record
# org = new Organization()


print("--finish--")
