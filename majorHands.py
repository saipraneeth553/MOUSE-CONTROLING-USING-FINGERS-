import cv2  # Can be installed using "pip install opencv-python"
import mediapipe as mp  # Can be installed using "pip install mediapipe"
import time
import math
import autopy
import pyautogui as p
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=False, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):    # Finds all hands in a frame
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):   # Fetches the position of hands
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            for x in self.results.multi_hand_landmarks:
                myHand = x

            #myHand = self.results.multi_hand_landmarks[1]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):    # Checks which fingers are up
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    # Finds distance between two fingers
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    frameR = 100
    width = 640             # Width of Camera
    height = 480
    screen_width, screen_height = autopy.screen.size()
    prev_x, prev_y = 0, 0   # Previous coordinates
    curr_x, curr_y = 0, 0
    smoothening = 8
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        sucess, img = cap.read()

        img = detector.findHands(img)
        print(img)

        lmlist, bbox = detector.findPosition(img)
        if len(lmlist) != 0:
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]

            fingers = detector.fingersUp()      # Checking if fingers are upwards
            cv2.rectangle(img, (frameR, frameR), (width - frameR, height -
                          frameR), (255, 0, 255), 2)   # Creating boundary box
            if fingers[4] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[1] == 1 and fingers[0]==0:
                x3 = np.interp(x1, (frameR, width-frameR), (0, screen_width))
                y3 = np.interp(y1, (frameR, height-frameR), (0, screen_height))
                curr_x = prev_x + (x3 - prev_x)/smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening
               # autopy.mouse.move(screen_width - curr_x, curr_y)
                p.scroll(200)
                cv2.waitKey(1000)
            if fingers[4] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[1] == 1 and fingers[0]==1:
                x3 = np.interp(x1, (frameR, width-frameR), (0, screen_width))
                y3 = np.interp(y1, (frameR, height-frameR), (0, screen_height))
                curr_x = prev_x + (x3 - prev_x)/smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening
               # autopy.mouse.move(screen_width - curr_x, curr_y)
                p.scroll(-200)
                cv2.waitKey(1000)
                
            if fingers[1] == 1 and fingers[2] == 0:
                # If fore finger is up and middle finger is down
                x3 = np.interp(x1, (frameR, width-frameR), (0, screen_width))
                y3 = np.interp(y1, (frameR, height-frameR), (0, screen_height))

                curr_x = prev_x + (x3 - prev_x)/smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening

                autopy.mouse.move(screen_width - curr_x,
                                  curr_y)    # Moving the cursor
                cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y
        

            # If fore finger & middle finger both are up
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                p.click(button='left')
                cv2.waitKey(1000)
            # if length < 40:     # If both fingers are really close to each other
                #cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                # autopy.mouse.click()    # Perform Click
                # print("ghhg")

            # If fore finger & middle finger both are up
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3]==1 and fingers[4]==0:
                length, img, lineInfo = detector.findDistance(8, 12, img)

               # if length < 40:     # If both fingers are really close to each other
                #cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                # autopy.mouse.click()    # Perform Click
                # print("ghhg")
                p.click(button='right')
                cv2.waitKey(1000)

                # lcv2.waitKey(1)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
