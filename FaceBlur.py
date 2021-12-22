import cv2
import numpy as np
cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.05,)
    for (x,y,w,h) in faces:
        blurredRegion = frame[y:y+h,x:x+w]
        blur = cv2.GaussianBlur(blurredRegion,(99,99),0)
        frame[y:y+h,x:x+w] = blur
    cv2.imshow("Zakaria Young", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
