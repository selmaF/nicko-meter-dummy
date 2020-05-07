import dlib
import cv2
import time
import numpy as np
#import imutils

face_cascade = cv2.CascadeClassifier('../assets/haarcascade_frontalface_default.xml')

landmark_model = dlib.shape_predictor('../assets/shape_predictor_68_face_landmarks.dat')

# for saving the video in output.avi from https://stackoverrun.com/de/q/11945245
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('NickenSchuetteln.mp4')
counter = 1
x0 = 0
y0 = 0
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(200, 200))


    for (x, y, w, h) in faces:
        counter = counter + 1
        face = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #(mx, my) = (int((x + w/2)), int((y + h/2)))
        #cv2.circle(frame, (mx,my), 1, (0, 255, 0), 6)
        print(counter)
        #if (counter % 2) == 0:
        #    print("mx " , mx, '\t x0', x0)
        #    x0 = mx
        shape = landmark_model(face, dlib.rectangle(0, 0, face.shape[0], face.shape[1]))

        # show landmarks of nose, left and right cheek, median eyebrow (up) and chin (down)
        (nx, ny) = (int(shape.part(30).x), int(shape.part(30).y))
        (lx, ly) = (int(shape.part(1).x), int(shape.part(1).y))
        (rx, ry) = (int(shape.part(15).x), int(shape.part(15).y))
        (rux, ruy) = (int(shape.part(22).x), int(shape.part(22).y))
        (lux, luy) = (int(shape.part(21).x), int(shape.part(21).y))
        (dx, dy) = (int(shape.part(8).x), int(shape.part(8).y))

        cv2.circle(frame, (x + nx, y + ny), 1, (0, 255, 0), 6)
        cv2.circle(frame, (x + lx, y + ly), 1, (255, 0, 0), 6)
        cv2.circle(frame, (x + rx, y + ry), 1, (255, 0, 0), 6)
        cv2.circle(frame, (x + rux, y + ruy), 1, (0, 0, 255), 6)
        cv2.circle(frame, (x + lux, y + luy), 1, (0, 0, 255), 6)
        cv2.circle(frame, (x + dx, y + dy), 1, (0, 0, 255), 6)

        video_writer.write(frame)  # Write the video to the file system
        # calculate difference in distance
        #distance_left = nx - lx
        #distance_right = rx - nx
        #print('left', distance_left, 'right', distance_right)
        #print('hight', y)
        #for i in range(0, 68):
        #    (cx, cy) = (int(shape.part(i).x), int(shape.part(i).y))
        #   cv2.circle(frame, (x + cx, y + cy), 1, (0, 255, 0), 6)


    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)
    if k == 5000:
        break


# When everything done, release the capture
cv2.destroyAllWindows()
video_writer.release()
cap.release()
