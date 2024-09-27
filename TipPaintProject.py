import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)  
cap.set(4, frameHeight) 
cap.set(10, 150)  


#replace this with the hue,val,sat of the color that you want e.g fingertip na may marking na orange, takip ng marker na kulay pink (use colorpicker.py)
myColor = [0, 81, 255, 179, 255, 255]
myColorValue = [51, 153, 255]

myPoints = []

def findColor(img, myColor, imgResult):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(myColor[0:3])
    upper = np.array(myColor[3:6])
    mask = cv2.inRange(imgHSV, lower, upper)
    x, y = getContours(mask, imgResult)
    if x != 0 and y != 0:
        myPoints.append([x, y])
    cv2.imshow("Mask", mask)
    return imgResult

def getContours(img, imgResult):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgResult, [cnt], -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y

def drawOnCanvas(myPoints, myColorValue):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValue, cv2.FILLED)

while True:
    success, img = cap.read()
    if not success:
        break
    imgResult = img.copy()

    imgResult = findColor(img, myColor, imgResult)
    
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValue)
    
    cv2.imshow("Result", imgResult)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
