import cv2
import numpy as np
from face_recognition import *
import os
from datetime import datetime

path='Images'
images=[]
classNames=[]

mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curimg=cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)
print(os.path.splitext(mylist[0]))

                                                    #''' ENCODING FUNCTION'''

def funcEncode(images):
    encodedImages=[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode= face_encodings(img)[0]
        encodedImages.append(encode)
    return encodedImages

encodingListKnown = funcEncode(images)
print('Encoded')



''' ATTENDENCE FUNCTION '''

def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        nameList=[]
        myDataList = f.readlines()

        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dateStr=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateStr}')
        
        #print(myDataList)
        



#   GETTING THE IMAGE FROM WEBCAM

cap=cv2.VideoCapture(0) # initialize webcam 0 is the id give to it

while True:
    success, img=cap.read()
    imgS=cv2.resize(img, (0,0), None, 0.25,0.25) # here we resize the image so that its easy to render it ( resizing to 1/4 of the size)
    imgS=cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame= face_locations(imgS)
    encodeCurFrame = face_encodings(imgS,facesCurFrame)  # passing the current frame so that it can get the location of which face to encode in case of multiple images

    for encodeFace,faceLoc in zip(encodeCurFrame, facesCurFrame):

        matches = compare_faces(encodingListKnown,encodeFace)
        faceDist= face_distance(encodingListKnown,encodeFace)
        #print(f'Matches={matches}  FaceDist={faceDist}')

        matchIndex= np.argmin(faceDist)

        if matches[matchIndex]:  # if true
            name=classNames[matchIndex] #get the name at this index
           # print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = (y1) *4,(x2) *4,(y2) *4,(x1) *4 # multiplying by 4 because we scales down our image to 1/4 and these are the scaled down measurements

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.rectangle(img,(x1,y2 -35),(x2,y2),(0,255,255),cv2.FILLED) # |__| above it gets the bound and fills it
            cv2.putText(img,name,(x1 +6,y2 -6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)
            


    cv2.imshow('Webcam',img)
    #cv2.waitKey(1)
    if cv2.waitKey(1) == 13:
        cap.release()
