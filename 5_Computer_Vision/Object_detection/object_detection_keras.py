from keras.models import load_model
import numpy as np
import cv2

# Load the pre-trained SSD MobileNet model.
model = load_model("ssd_mobilenet_v2/keras.h5")

# common objects detected by the model
LABELS=["person"]#, "car", "motorcycle", "bus", "bicycle", "airplane", "train"]

"""
Detect objects in a given frame (image or video frame) 
    using the loaded SSD MobileNet model.

Arg:
    frame (numpy.ndarray) : NumPy array with the captured frame.

Returns:
    boxes (numpy.ndarray)   : Bounding box coordinates of detected objects.
    classes (numpy.ndarray) : Class indices of the detected objects.
    scores (numpy.ndarray)  : Confidence scores of the detected objects.
"""
def detect_objects(frame):
    
    # -- preprocess frame ------------------------------------------------------    
    input_frame =cv2.resize(frame, (300, 300))  # Resize to the input shape of the model
    input_frame =np.expand_dims(input_frame, axis=0)  # Add batch dimension
    input_frame =input_frame.astype(np.float32) / 255.0  # Normalize the input

    

    # -- detection -------------------------------------------------------------
    # object detection with the model
    detections =model.predict(input_frame)

    boxes   =detections[0]['detection_boxes'][0]  
    classes =detections[0]['detection_classes'][0].astype(int)
    scores  =detections[0]['detection_scores'][0]

    return boxes, classes, scores
