#Object Detection From an Image

import cv2
import numpy as np

# Load YOLO pre-trained weights and configuration files
yolo_weights = "yolov3.weights"
yolo_config = "yolov3.cfg"
labels_file = "coco.names"

# Load class names
with open(labels_file, "r") as f:
    class_names = f.read().strip().split("\n")

# Load the image
image_path = 'image.png'  # Replace with your image path
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Initialize the YOLO model
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getUnconnectedOutLayersNames()

# Run forward pass
detections = net.forward(layer_names)

# Initialize lists for NMS
boxes = []
confidences = []
class_ids = []

# Process detections
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            box = obj[0:4] * np.array([width, height, width, height])
            center_x, center_y, w, h = box.astype("int")

            # Calculate top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Draw the final bounding boxes
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow("Car Detection", image)

# Wait for a key press and close the display window
cv2.waitKey(0)
cv2.destroyAllWindows()