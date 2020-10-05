import cv2
import numpy as np
import face_recognition

#                                                               First step :- to load images

imgElon = face_recognition.load_image_file('ImageBasics/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)


imgTest= face_recognition.load_image_file('ImageBasics/elon_musk_test.jpg')
imgTest= cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#                                                           step number 2 :-  Finding faces in the image and getting their encodings as well

faceloc= face_recognition.face_locations(imgElon)[0] # we take the first element because it returns a list of tuples (top, right ,bottom,left) in order
                    ##print(faceloc)
encodeElon=face_recognition.face_encodings(imgElon)[0] # this gives us the encodings
                    #print(encodeElon) # returns a numpy array of encodings

                    #print(faceloc)
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),2) # this draws a rectangle around the face.


'''' Test image face '''

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]), (facelocTest[1],facelocTest[2]),(0,255,0),2)


# LOADING BILL GATES IMAGE
imgBill=face_recognition.load_image_file('ImageBasics/Bill Gates.jpg')
imgBill=cv2.cvtColor(imgBill,cv2.COLOR_BGR2RGB)
faceLocBill=face_recognition.face_locations(imgBill)[0]
encodeBill=face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill,(faceLocBill[3],faceLocBill[0]),(faceLocBill[1], faceLocBill[2]),(0,255,0),2)


''' STEP3 WE WILL COMPARE THE IMAGES ( THIS USES LINEAR SVM IM BACKEND TO COMPARE)'''

result=face_recognition.compare_faces([encodeElon,encodeBill],encodeTest)
faceDist=face_recognition.face_distance([encodeElon,encodeBill],encodeTest) # lower the distance more is the match
print(result,faceDist)
cv2.putText(imgTest,f'{result} Elon-{round(faceDist[0],2)} Bill-{round(faceDist[1],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Elon Musk',imgElon)
cv2.namedWindow("Elon Musk", cv2.WINDOW_NORMAL)
cv2.waitKey(0)

cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)

cv2.imshow("Bill Gates",imgBill)
cv2.waitKey(0)

