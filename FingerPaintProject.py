#paint project
import cv2
import numpy as np
import time
import os 
import HandTrackingModule as htm

folderpath = "header"
myList = os.listdir(folderpath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderpath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
header = overlayList[0]  
drawColor = (255, 0, 0)

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 520)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((520, 1080, 3), np.uint8)

brushThickness = 15
eraserThickness = 45

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 1. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 2. Check which fingers are up
        fingers = detector.fingersUp()

        # 3. If selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            print("Selection Mode")
            xp, yp = 0, 0

            # Check the click
            if y1 < 125:
                if 55 < x1 < 250:
                    header = overlayList[0]
                    drawColor = (255, 0, 0)
                elif 358 < x1 < 538:
                    header = overlayList[3]  # red
                    drawColor = (0, 0, 255)
                elif 608 < x1 < 798:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 838 < x1 < 1080:
                    header = overlayList[1]  # eraser
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 30), (x2, y2 + 30), drawColor, cv2.FILLED)

        # 4. If drawing mode - index finger is up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    # Resize imgCanvas to match img dimensions
    imgCanvas = cv2.resize(imgCanvas, (img.shape[1], img.shape[0]))

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)

    # Resize imgInv to match img dimensions
    imgInv = cv2.resize(imgInv, (img.shape[1], img.shape[0]))
    ImgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
 
    # Ensure img and imgCanvas have the same shape
    img = cv2.bitwise_and(img, ImgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Overlay header
    header_resized = cv2.resize(header, (img.shape[1], 100))  # Resize to the width of the frame
    img[0:100, 0:img.shape[1]] = header_resized

    # Display the resulting frame
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inverse", ImgInv)
    cv2.waitKey(1)


