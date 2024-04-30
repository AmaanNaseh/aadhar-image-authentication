import cv2, imutils, time, face_recognition
from PIL import Image
import numpy as np
import streamlit as st

cam = cv2.VideoCapture(0)

text = "Press S to click image"
color = (0, 0, 255)
org = (30, 50)
text2 = ""

st.title("Live Identity Authentication")
empty = st.empty()

save_button = st.button("Save")
stop_button_pressed = st.button("Stop")

################################ Capturing Image ################################

while cam.isOpened() and not stop_button_pressed:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=550, height=550)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    key = cv2.waitKey(10)
    
    if key == ord("s") or save_button:
        text = 'Image captured Successfully'
        color = (0, 255, 0)
        org = (25, 50)
        text2 = "Please Exit by pressing Escape"
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite("image.jpg", frame)    
        print(text)

    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_COMPLEX, 1, color)
    
    cv2.putText(frame, text2, (25,100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255))
    
    #cv2.imshow("Frame", frame)

    empty.image(frame, channels="RGB")

    if cv2.waitKey(1) & 0xFF == 27 or stop_button_pressed:
        break

cam.release()
cv2.destroyAllWindows()


################################ jpg to webp ################################

# im_source = Image.open("aadhar_photo.jpg").convert("RGB") # give path of aadhar photo
# im_source.save("aadhar_photo.webp", "webp")
# print("Image converted to webp successfully")

im_live = Image.open("image.jpg").convert("RGB") # give path of live image
im_live.save("image.webp", "webp")
print("Image converted to webp successfully")


################################ Face recognition ################################

source_img = cv2.imread("Messi1.webp")
rgb_source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
img_encoding1 = face_recognition.face_encodings(rgb_source_img)[0]

live_img = cv2.imread("image.webp")
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