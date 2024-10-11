import pybullet as pb



"""
Function to navigate robot based on detected objects

Args:
    robot_id (int)    : Robot ID. 
    boxes (float[])   : List of bounding boxes around detected objects.
    classes (int[])   : List of classes for the detected objects.
    scores (float[])  : List of scores for the detected objects.
    threshold (float) : A threshold to determine which detections are valid
"""
def navigate_robot(robot_id, boxes, classes, scores, threshold=0.5):
    
    # iterates through the detected bounding boxes and their corresponding scores.
    for i, box in enumerate(boxes):
        
        # checks if it is a valid object
        if scores[i]>threshold:
            # Object is close to the robot
            if box[0]<0.3: turn_robot_right(robot_id)
            else: move_robot_forward(robot_id)

def move_robot_forward(robot_id):
    # Command the robot to move forward
    pb.setJointMotorControl2(robot_id, 0, pb.VELOCITY_CONTROL, targetVelocity=1)

def turn_robot_right(robot_id):
    # Command the robot to turn right
    pb.setJointMotorControl2(robot_id, 0, pb.VELOCITY_CONTROL, targetVelocity=-1)
