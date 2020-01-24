# Optical Mark Recognition (OMR) for reading the assessment scoring forms
# created by: Peter Dobbs

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from pdf2image import convert_from_path
from fhir.resources.patient import Patient
from fhir.resources.humanname import HumanName
from fhir.resources.encounter import Encounter
from fhir.resources.documentreference import DocumentReference
from fhir.resources.observation import Observation

# constants
RED = (0, 0, 255)
BLUE = (255, 0, 0)
NUM_ANSWER_OPTIONS = 4
W_THRESH = 0.0
H_THRESH = 0.0

# variables
subtotal = 0
questionOffset = 0  # useful for multi-page questionaire
answers = {}


def score_assessment_form(file_string, page_num, wThresh, hThresh):
    global questionOffset, subtotal, answers
    file_img = cv2.imread(file_string)

    # restrict the contour detection to a scoring_region
    (rows, columns) = file_img.shape[:2]
    widthThreshold = int(columns*wThresh)
    heightThreshold = int(rows*hThresh)
    region = file_img[heightThreshold:rows, widthThreshold:columns]

    # scale, gray, and threshold the image
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    bwImage = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imshow('black and white image', bwImage)
    # cv2.waitKey(0)

    # detect scoring boxes (bubbles in this case)
    # find contours
    # -- this isn't exclusive enough -- It's picking up letters like 'M' and 'O'
    # -- this also really sucks on bubbles filled by hand
    cnts, hierarchy = cv2.findContours(bwImage, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    print("Number of detected contours:", len(cnts))

    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if w >= 30 and h >= 30 and aspect_ratio >= 0.80 and aspect_ratio <= 1.2:
            questionCnts.append(c)

    questionCnts = contours.sort_contours(
        questionCnts, method="top-to-bottom")[0]
    print("Number of question-related contours:", len(questionCnts))

    cv2.drawContours(region, questionCnts, -1, BLUE, 6)
    cv2.imshow('black and white with overlayed contours', region)
    # cv2.imwrite('detected_contours.png', region)
    # cv2.waitKey(0)

    # arrange questions into groups
    questionGroups = np.arange(0, len(questionCnts), NUM_ANSWER_OPTIONS)

    # loop through answer groups
    for (q, i) in enumerate(questionGroups):
        bubbled = (0, 0)
        expectedBubbleFill = 0

        # order the answer group from left-to-right
        cnts = contours.sort_contours(
            questionCnts[i:i + NUM_ANSWER_OPTIONS])[0]

        # find filled-in bubble
        for (j, c) in enumerate(cnts):
            mask = np.zeros(bwImage.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            mask = cv2.bitwise_and(bwImage, bwImage, mask=mask)
            total = cv2.countNonZero(mask)
            expectedBubbleFill += total

            if bubbled == (0, 0) or total > bubbled[0]:
                bubbled = (total, j)

        expectedBubbleFill /= NUM_ANSWER_OPTIONS

        if(bubbled[0] > expectedBubbleFill*1.2):   # threshold for `filled-in bubble`
            answers[questionOffset+q+1] = (bubbled[1]+1)
            cv2.drawContours(region, [cnts[bubbled[1]]], -1, RED, 5)
            subtotal += (bubbled[1]+1)
        else:
            print('Missing answer detected at question ', questionOffset+q+1)
            answers[questionOffset+q+1] = None

    questionOffset = q+1

    cv2.imshow("Scoring region", region)
    cv2.imwrite(
        'temp/detected_contours_scored{}.png'.format(page_num), region)
    # cv2.waitKey(0)


print("--start--")
print("--PRESETS--")
print('')

# pdf_file = "assessment-filled.pdf"
# pages = convert_from_path(pdf_file)
# count = 0

# for page in pages:
#     page_file = "temp/pg"+str(count)+".png"
#     page.save(page_file, "PNG")
#     count = count+1

# for page_num in range(count):
#     page_file = "temp/pg"+str(page_num)+".png"
#     score_assessment_form(page_file, page_num,
#                           wThresh=W_THRESH, hThresh=H_THRESH)

# print("Raw Score:", subtotal)


# creating FHIR resource for patient record
pat = Patient({
    "id": "1",
    "name": [
        {
            "given": ["John"],
            "family": "Doe"
        }
    ],
    "birthDate": "2019-01-24"
})
print(pat)

enc = Encounter()
enc.subject = pat

docref_json = {
    "resourceType": "DocumentReference",
}
doc = DocumentReference()
obs = Observation()


print("--finish--")
