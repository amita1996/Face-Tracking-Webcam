import cv2
import face_recognition as fr
import pickle
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep , perf_counter
import numpy as np
WIDTH = 640
HEIGHT = 480
factory = PiGPIOFactory()
servoy = Servo(23,pin_factory = factory,min_pulse_width = 0.5/1000,max_pulse_width = 2/1000)

CURRENT_X = 0
CURRENT_Y = -0.25
Y_COOLDOWN = perf_counter()
X_COOLDOWN = perf_counter()
servoy.value = CURRENT_Y

#Loading my image encoding
def load_encoding():
    file = open('encoding_list','rb')
    data = pickle.load(file)
    file.close()
    return data


def find_target(cap,encoding_list,face_cascade):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor= 1.05,minNeighbors = 10, minSize = (50,50))
            for face in faces:
                resized_frame = cv2.resize(frame,(160,120),interpolation=cv2.INTER_AREA)
                x,y,w,h = [int(num/4) for num in face]
                frame_face_encoding = fr.face_encodings(cv2.cvtColor(resized_frame,cv2.COLOR_BGR2RGB),[(y,x+w,y+h,x)])
                if frame_face_encoding:
                    matches = fr.compare_faces(encoding_list, frame_face_encoding[0],tolerance=0.55)
                    x,y,w,h = face
                    if True in matches:
                        cv2.rectangle(frame, (x,y), (x+w,y+h), color=(0, 255, 0), thickness=2)
                        cv2.putText(frame, 'Amit', (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (36, 255, 12), 2)
                        cv2.imshow('Frame', frame)
                        return tuple(face), frame

                    else:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),color = (0,0,255), thickness=2)
                        cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                return None
        else:
            return None


def track_face(cap,args):
    MOVE_THRESHOLD_X = 50
    MOVE_THRESHOLD_Y = 70
    tracker = cv2.TrackerCSRT_create()
    bbox, frame = args
    _ = tracker.init(frame,bbox)
    while cap.isOpened():
        ret, frame = cap.read()
        ret,bbox = tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]),int(bbox[1]))
            p2 = (int(bbox[0]) + int(bbox[2]), int(bbox[1] + int(bbox[3])))
            cv2.rectangle(frame,p1,p2,color = (0,255,0) , thickness=2)
            cv2.putText(frame, 'Amit', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.imshow('Frame', frame)
            x_center = int(bbox[0] + bbox[2]/2)
            y_center = int(bbox[1] + bbox[3]/2)
            if abs(x_center - WIDTH/2) > MOVE_THRESHOLD_X:
                move_x(x_center)
            if abs(y_center - HEIGHT/2) > MOVE_THRESHOLD_Y:
                move_y(y_center)
        else:
            return -1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            return False

def move_x(x_center):
    print("move x")
    if x_center > WIDTH/2:
        pass
    else:
        #move right
        pass


def move_y(y_center):
    global CURRENT_Y
    global Y_COOLDOWN
    global servoy
    if perf_counter() - Y_COOLDOWN > 2:
        if y_center > HEIGHT/2 and CURRENT_Y < 0.9:
            servoy.value = CURRENT_Y + 0.1
        elif CURRENT_Y > -0.9:
            servoy.value = CURRENT_Y - 0.1
        Y_COOLDOWN = perf_counter()
        CURRENT_Y = servoy.value


face_cascade = cv2.CascadeClassifier(r'/usr/local/lib/python3.7/dist-packages/cv2/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

encoding_list = load_encoding()

while True:
    target = find_target(cap,encoding_list,face_cascade)
    if not target:
        break
    tracking = track_face(cap,target)
    if not tracking:
        break

servoy.value = None
cap.release()
cv2.destroyAllWindows()