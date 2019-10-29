#import cv2

#cascPath = "haarcascade_frontalface_default.xml"

#faceCascade = cv2.CascadeClassifier(cascPath)

#cap = cv2.VideoCapture(0)

#if (cap.isOpened() == False): #lines mean: If camera is not opend then print
   # {
   #  print ("unable to read camera output")
   #  }
    
#while(True):
    
    #ret, frame = cap.read()
    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #if ret == True: #getting camera input
       #faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
       #minSize=(1,1),
       #flags = cv2.CASCADE_SCALE_IMAGE
       #)
       
       #for (x, y, w, h) in faces: cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
       
      # cv2.imshow('My frame', frame)
       
       #if cv2.waitKey(1) & 0xFF == ord('q'):
       #    break
    #else:
         #break
     
#cap.release()
#cv2.destroyAllWindows()
         
         
         
import cv2
import pickle

face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
cap=cv2.VideoCapture(0)

while(True):
    #capture frame by frame
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces: 
        print(x,y,w,h)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #recognize deep learned model
        id_, conf = recognizer.predict(roi_gray)
        if conf>=4 and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        img_item="my-image.png"
        #creating rectangle
        color = (255, 0, 0) #BGR 0-255 
        stroke=2
        end_cord_x=x+w#width
        end_cord_y=y+h#height
        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y),color,stroke)
        cv2.imwrite(img_item,roi_color)
        
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()