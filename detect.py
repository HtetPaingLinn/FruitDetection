import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
# Create a function to preprocess an image for the model
def preprocess_image(image):
    image = cv2.resize(image, (75, 75))
    return image

# Create a function to get class predictions and confidence
def predict_image_classes(image, model_path, class_labels):
    # Load the pre-trained model
    model = load_model(model_path)

    # Get the target size from the model input shape
    target_size = tuple(model.input.shape[1:3])

    # Perform inference
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    confidence = prediction[predicted_class_index]



    return predicted_class, confidence

# Open the laptop camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    model_path = r"C:\Users\olumi\Desktop\testing\resnet50model.hdf5"
    class_labels = [0,1] 

    preprocessed_image = preprocess_image(frame)
    class_name, confidence = predict_image_classes(preprocessed_image, model_path, class_labels)

    cv2.putText(frame, f'Class: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Image Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
