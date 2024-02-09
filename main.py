import cv2.cv2 as cv2
import numpy as np
import os
from datetime import datetime
import dlib
import face_recognition

path = 'resources/image_attendance'
images_To_Train = []
images_To_Train_names = []
face_train_list = os.listdir(path)
for face_train_list_elements in face_train_list:
    curFace_train = cv2.imread(f'{path}/{face_train_list_elements}')
    images_To_Train.append(curFace_train)
    images_To_Train_names.append(os.path.splitext(face_train_list_elements)[0])

def getEncodings(images):
    encoding_list=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encoding_list.append(encode)
    return encoding_list
encodings_known_list =getEncodings(images_To_Train)
print('Encoding Complete')
def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



#takes webcam as input
cap = cv2.VideoCapture(0)
cap.set(3,680)
cap.set(4,460)

#
while True:
    success, img_frame = cap.read()
    img_frame_RGB = cv2.resize(img_frame,(0,0),None,0.25,0.25)
    img_frame_RGB = cv2.cvtColor(img_frame,cv2.COLOR_BGR2RGB)

    img_frame_locations = face_recognition.face_locations(img_frame_RGB)
    img_frame_encodings = face_recognition.face_encodings(img_frame_RGB,img_frame_locations)

    for encodes_per_img_frame,location_per_img_frame in zip(img_frame_encodings,img_frame_locations):
        matches = face_recognition.compare_faces(encodings_known_list,encodes_per_img_frame)
        face_distance = face_recognition.face_distance(encodings_known_list,encodes_per_img_frame)
        print(face_distance)
        matchIndex = np.argmin(face_distance)#face distance of matching face is lowest so the matching image will be
                                             #at the first index

        if matches[matchIndex]:
            match_name = images_To_Train_names[matchIndex].upper()
            markAttendance(match_name)
            print(match_name)
            y1,x2,y2,x1 = location_per_img_frame
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 #multiplying by 4 bcz we have scaled down our input by 1/4th
                                                 #to get the real posn of the input for bounding box
            cv2.rectangle(img_frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img_frame, (x1, y2-35), (x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img_frame,match_name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)



    cv2.imshow('Alpha', img_frame)

    key = cv2.waitKey(1)
    if key==27:
        break
