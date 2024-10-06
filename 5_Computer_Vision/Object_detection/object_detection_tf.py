import tensorflow as tf



# load the pre-trained SSD MobileNet model. TensorFlow Hub
# https://tfhub.dev/tensorflow/ssd_mobilenet_v2

model=tf.saved_model.load("ssd_mobilenet_v2/")

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
    # convert the frame to a tensor
    input_tensor =tf.convert_to_tensor(frame)    
    # convert to uint8 (expected input)     
    input_tensor =tf.cast(input_tensor, tf.uint8) 
    
    
    
    # -- detection -------------------------------------------------------------    
    # object detection with the model
    detections=model(input_tensor[tf.newaxis])
    
    boxes   =detections['detection_boxes'][0].numpy()
    classes =detections['detection_classes'][0].numpy().astype(int)
    scores  =detections['detection_scores'][0].numpy()

    return boxes, classes, scores
