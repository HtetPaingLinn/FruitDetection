import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoFrame

# Define a video processor class for the YOLO model
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 tracking on the frame
        results = self.model.track(img, persist=True)
        annotated_frame = results[0].plot()
        annotated_array = np.array(annotated_frame)

        return VideoFrame.from_ndarray(annotated_array, format="bgr24")

def main():
    st.title("Fruit Tracking and Inspection")

    # Define a function to get the model directory based on the selected fruit
    def get_model_directory(selected_fruit):
        fruit_model_mapping = {
            "Tomato": "best1.pt",
            "Banana": "best33.pt",
            "Orange": "best2.pt",
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
    st.sidebar.info("This app uses YOLOv8 to track and inspect fruits in real-time.")

    # Create a video streamer using streamlit-webrtc
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=lambda: YOLOVideoProcessor(model),
        media_stream_constraints={"video": True, "audio": False},
    )

if __name__ == "__main__":
    main()
