import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import urllib.request

def main():
    st.title("Fruit Tracking and Inspection")
    cap_url = "http://192.168.195.55/640x480.jpg"
    # Open the camera
    cap = cv2.VideoCapture(0)
    def get_model_directory(selected_fruit):
        fruit_model_mapping = {
            "Tomato": "best1.pt",
            "Banana": "best33.pt",
            "Orange": "best22.pt",
        }
        return fruit_model_mapping.get(selected_fruit, "default_model_directory")

    st.sidebar.header("Fruit Selection")
    selected_fruit = st.sidebar.selectbox("Select a fruit", ["Tomato", "Banana", "Orange"])

    # Get the model directory based on the selected fruit
    model_directory = get_model_directory(selected_fruit)

    # Load the YOLO model using the selected model directory
    model = YOLO(model_directory)

    # Display information about the selected fruit
    st.sidebar.info(f"Selected Fruit: {selected_fruit}")
    st.sidebar.info("This app uses YOLOv8 to track and inspects fruits in real-time.")

    # Placeholder for video display
    video_placeholder = st.empty()
    if st.button("start Tracking"):      
        # Loop through the camera frames
        while True:
            try:
                img_resp=urllib.request.urlopen(cap_url)
                imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
                #ret, frame = cap.read()
                frame = cv2.imdecode(imgnp,-1)
                # Read a frame from the camera
                #success, frame = cap.read()

         
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True)
                annotated_frame = results[0].plot()
                annotated_array = np.array(annotated_frame)

                # Display the annotated frame in the main area
                video_placeholder.image(annotated_array, channels="BGR")
            except:
                st.warning("Unable to capture video. Please check your camera.")
                break

        # Release the video capture object
        cap.release()
    elif st.button("Stop Tracking"):
        video_placeholder.info("Tracking Disabled")
                 
if __name__ == "__main__":
    main()
