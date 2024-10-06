import cv2

from object_detection/tf import detect_objects

def main():

    # initialize video capture
    capture=cv2.VideoCapture(0)

    # colors
    green_RGB=(0, 255, 0)
    red_RGB=(255, 0, 0)

    print('\n\n-- Terminates the execution by pressing "q" key -- \n\n')
    try:
        while True:
        # ----------------------------------------------------------------------            
            # capture frame-by-frame from the camera.
            # ok (bool) : frame is succesfully captured.
            ok, frame =capture.read()
            if not ok: break

            # detect the objects in the given frame.
            boxes, classes, scores =detect_objects(frame)
            
            
            # display the video with the detected objects.
            for i, box in enumerate(boxes):
                # only processes boxes with a 0.5 score or greater 
                if scores[i]>0.5:
                    # box coordinates
                    # top, left, bot, right
                    ymin, xmin, ymax, xmax =box
                    
                    # dimensions (height, width)
                    height, width, _ =frame.shape
                    # converting normalized coordinates
                    xmin=int(xmin*width)
                    xmax=int(xmax*width)
                    ymin=int(ymin*height)
                    ymax=int(ymax*height)
                    
                    # detected object label
                    label=f"Object {classes[i]}"
                    
                    # drawing the green rectangle
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), 
                                  green_RGB, 2)
                    cv2.putText(frame, label, (xmin, ymin-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, red_RGB, 2)

            # display the frame
            cv2.imshow('View', frame)

            # break loop with 'q' key
            if cv2.waitKey(1) & 0xFF==ord('q'): break

    
    finally: # resource cleanup and close simulation
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
