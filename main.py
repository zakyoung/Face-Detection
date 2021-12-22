import cv2
cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.05,6)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 10)
    cv2.imshow("Zakaria Young", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
