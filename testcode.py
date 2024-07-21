import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import urllib.request
# Load the YOLOv8 model
model = YOLO('best1.pt')

# Open the video file
video_path = "WhatsApp Video 2023-11-01 at 18.30.03_ac0c0f57.mp4"
cap = cv2.VideoCapture(0)
cap_url = "http://192.168.145.55/640x480.jpg"
# Loop through the video frames
while cap.isOpened():
    img_resp=urllib.request.urlopen(cap_url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    # Read a frame from the video
    frame = cv2.imdecode(imgnp,-1)

   
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    print(results[0])

    # Display the annotated frame
    cv2.imshow("Tomato Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()