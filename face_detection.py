import numpy as np
import cv2
import math

FACE_CLASSIFIER_FILE = 'text_data/haarcascade_frontalface.xml'
EYE_CLASSIFIER_FILE = 'text_data/haarcascade_eye.xml'

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result

def main():
    face_cascade = cv2.CascadeClassifier(FACE_CLASSIFIER_FILE)
    eye_cascade = cv2.CascadeClassifier(EYE_CLASSIFIER_FILE)

    img = cv2.imread('angle.jpg')
    small = cv2.resize(img, (0,0), fx=0.4, fy=0.4) 
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))
    for (x,y,w,h) in faces:
        small = cv2.rectangle(small,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        #roi_color = small[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #rotate image
        x1, y1, w1, h1 = eyes[0]
        x2, y2, w2, h2 = eyes[1]
        dy = (y2 - y1)
        dx = (x2 - x1)
        angle = (2+ math.atan(dy/dx) * 180) / 3.14
        rotated_image = gray
        rotated_image = rotateImage(gray, angle)



    cv2.imshow('img', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()