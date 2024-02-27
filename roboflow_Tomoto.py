'''
from roboflow import Roboflow

rf = Roboflow(api_key="hA7grH1nf9tiKCl2aYlP")
project = rf.workspace().project("tomotao_disease_prediction")
model = project.version(2).model

# infer on a local image
print(model.predict("/home/amma/Music/toma_mul_cascade/Tomato___Bacterial_spot/p/53824ab2-5655-4524-916b-9900ee0b949b___UF.GRC_BS_Lab Leaf 1159.JPG", confidence=19, overlap=50).json())
import cv2
import matplotlib.pyplot as plt
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="hA7grH1nf9tiKCl2aYlP")
project = rf.workspace().project("tomotao_disease_prediction")
model = project.version(2).model

# Path to the image
image_path = "/home/amma/Music/toma_mul_cascade/Tomato___Early_blight/p/3fa32b26-e7bb-493e-b7c2-c40e3f3df380___RS_Erly.B 9546.JPG"

# Perform prediction
predictions = model.predict(image_path, confidence=40, overlap=50).json()

# Load the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image)

# Plot bounding boxes
for prediction in predictions['predictions']:
    x, y, w, h = (
        prediction['x'],
        prediction['y'],
        prediction['width'],
        prediction['height']
    )
    confidence = prediction['confidence']
    
    # Draw bounding box
    rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    
    # Display confidence score
    plt.text(x, y - 5, f"{confidence:.2f}", color='r')

plt.show()'''
'''import cv2
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="hA7grH1nf9tiKCl2aYlP")
project = rf.workspace().project("tomotao_disease_prediction")
model = project.version(2).model

# Path to the image
image_path = "/home/amma/Music/toma_mul_cascade/Tomato___Bacterial_spot/p/53824ab2-5655-4524-916b-9900ee0b949b___UF.GRC_BS_Lab Leaf 1159.JPG"

# Perform prediction
predictions = model.predict(image_path, confidence=19, overlap=50).json()

# Load the image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Draw bounding boxes on the image
for prediction in predictions['predictions']:
    x, y, w, h = (
        int(prediction['x']),
        int(prediction['y']),
        int(prediction['width']),
        int(prediction['height'])
    )
    confidence = prediction['confidence']
    class_name = prediction['class']
    
    # Draw bounding box
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display class name and confidence
    text = f"{class_name}: {confidence:.2f}"
    cv2.putText(image_rgb, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Predictions', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import cv2
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="your API key")
project = rf.workspace().project("tomotao_disease_prediction")
model = project.version(2).model

# Open a connection to the webcam (use 0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform prediction
    predictions = model.predict(frame, confidence=43, overlap=50).json()

    # Draw bounding boxes on the frame
    for prediction in predictions['predictions']:
        x, y, w, h = (
            int(prediction['x']),
            int(prediction['y']),
            int(prediction['width']),
            int(prediction['height'])
        )
        confidence = prediction['confidence']
        class_name = prediction['class']

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display class name and confidence
        text = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Predictions', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

