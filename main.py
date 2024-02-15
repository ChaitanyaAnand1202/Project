import cv2
import numpy as np
import time

# Loading YOLO weights, configuration, and COCO names
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Loading video file
cap = cv2.VideoCapture('veh2.avi')

# Defining the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Defining video writer for MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
output = cv2.VideoWriter('result.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# Defining the two lines on the screen
line1_y = 100  # Adjust as needed
line2_y = 300  # Adjust as needed
line_distance = 10 # Adjust as needed(real life distance)

# Variable to store the start time
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Applying YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Drawing the lines on the screen
    cv2.line(frame, (0, line1_y), (frame.shape[1], line1_y), (0, 255, 0), 2)
    cv2.line(frame, (0, line2_y), (frame.shape[1], line2_y), (0, 255, 0), 2)

    for i, box in enumerate(boxes):
        x, y, w, h = box

        # Calculating the midpoint of the detected vehicle
        vehicle_midpoint = (x + w // 2, y + h // 2)

        # Checking if the vehicle crossed line1
        if y < line1_y and vehicle_midpoint[1] > line1_y:
            start_time = time.time()  # Record time at line1 crossing

        # Checking if the vehicle crossed line2
        if y < line2_y and vehicle_midpoint[1] > line2_y and start_time is not None:
            crossing_time = time.time()  # Record time at line2 crossing
            vehicle_speed = (line_distance / (crossing_time - start_time)) * 3.6 

            # Displaying vehicle speed on frame
            cv2.putText(frame, f'Speed: {vehicle_speed:.2f} km/h', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = str(classes[class_ids[i]])
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    

    cv2.imshow('Result', frame)
    output.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
