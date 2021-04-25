import cv2
import numpy as np
import face_recognition

imgA = face_recognition.load_image_file('ImagesBasic/Sakti.jpg')
imgA = cv2.cvtColor(imgA,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/Mukti.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgA)[0]
encodeA = face_recognition.face_encodings(imgA)[0]
cv2.rectangle(imgA,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeA],encodeTest)
faceDis = face_recognition.face_distance([encodeA],encodeTest)
print(results,faceDis)

cv2.imshow('Image 1',imgA)
cv2.imshow('Image 2',imgTest)
cv2.waitKey(0)