import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontface_default.xml")

our_image_color = cv2.imread("IMG_20230914_153024_009.jpg")
our_image_gray = cv2.cvtColor(our_image_color,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(our_image_gray,
scaleFactor = 1.05,
minNeighbors = 5
)

for x, y, w, h in faces:
    our_image_rect = cv2.rectangle(our_image_color, (x,y), (x+w,y+h), (0,255,0), 3)

cv2.imshow("Face Detection", our_image_rect)
cv2.waitKey(0)
cv2.destroyAllWindows
