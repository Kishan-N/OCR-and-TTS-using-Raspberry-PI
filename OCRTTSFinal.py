import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import os
from gtts import gTTS
from playsound import playsound
import os.path
from os import path
from pygame import mixer
pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'
mixer.init()

img='real.jpeg'

def OCRImage():
    

    myfile=open("test.txt","w")
    myfile.close()

    frame = cv2.imread(img)
    norm_frame = np.zeros((frame.shape[0], frame.shape[1]))
    frame = cv2.normalize(frame, norm_frame, 0, 255, cv2.NORM_MINMAX)
    frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)[1]
    frame = cv2.GaussianBlur(frame, (1, 1), 0)
    cv2.imshow('Picture', frame)
    cv2.waitKey(1)
    sctext=pytesseract.image_to_string(frame)

    myfile=open("test.txt","w")
    with open('test.txt', mode='a') as file:
        file.write(sctext)
    myfile.close()

    language='en'

    if((os.stat("test.txt").st_size == 0)==False):
        myfile=open("test.txt","r")
        contents=myfile.read()
        myfile.close()
        obj=gTTS(text=contents, lang=language, slow=False)
        obj.save("sample.mp3")
    if(path.exists('sample.mp3')==True):
        mixer.music.load("sample.mp3")
        mixer.music.set_volume(1)
        mixer.music.play()

def OCRRealTime():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        d = pytesseract.image_to_data(frame, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if float(d['conf'][i]) > 60:
                (text, x, y, w, h) = (d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                # don't show empty text
                if text and text.strip() != "":
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    frame = cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    print(text)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

while True:
    print("Press 1 to convert text from image to speech." )
    print("Press 2 to Recognize text in real time." )
    option = input("Which option do you choose (1 - 2): ")
    print("\n")
    if option == '1':
        OCRImage()
    elif option == '2':
        OCRRealTime()
    else:
        print("Thank you for using the the OCR program")
        break
