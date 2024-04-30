import cv2, imutils, time, face_recognition
from PIL import Image
import numpy as np

cam = cv2.VideoCapture(0)

text = "Press S to click image"
color = (0, 0, 255)
org = (30, 50)
text2 = ""

################################ Capturing Image ################################

while cam.isOpened():
    _, frame = cam.read()
    frame = imutils.resize(frame, width=550, height=550)
    
    key = cv2.waitKey(10)
    
    if key == ord("s"):
        text = 'Image captured Successfully'
        color = (0, 255, 0)
        org = (25, 50)
        text2 = "Please Exit by pressing Escape"

        cv2.imwrite("image.jpg", frame)    
        print(text)

    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_COMPLEX, 1, color)
    
    cv2.putText(frame, text2, (25,100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255))
    
    cv2.imshow("Frame", frame)

    if key == 27:
        break

################################ jpg to webp ################################

# im_source = Image.open("aadhar_photo.jpg").convert("RGB") # give path of aadhar jpg photo
# im_source.save("aadhar_photo.webp", "webp")
# print("Source Image converted to webp successfully")

im_live = Image.open("image.jpg").convert("RGB") # give path of live jpg image
im_live.save("image.webp", "webp")
print("Live Image converted to webp successfully")


################################ Face recognition ################################

source_img = cv2.imread("Messi1.webp") #give path of converted webp image
rgb_source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
img_encoding1 = face_recognition.face_encodings(rgb_source_img)[0]

live_img = cv2.imread("image.webp") #give path of converted webp image
rgb_live_img = cv2.cvtColor(live_img, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_live_img)[0]

result = face_recognition.compare_faces([img_encoding1], img_encoding2)
print("Result: ", result)

source_img = imutils.resize(source_img, width=550, height=550)

#cv2.imshow("Reference Image", source_img)
#cv2.imshow("Live Image", live_img)

display = np.row_stack((source_img, live_img))

cv2.imshow("Identity Recognition", display)
cv2.waitKey(0)

cam.release()
cv2.destroyAllWindows()