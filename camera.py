import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt



# Load the YOLOv10 model
model = YOLO('D:\RPSLO object detection\Robot-Pembersih-Sampah-Laut-Otonom/botolyolov10.pt')
model.to('cuda')
# Open the video capture device (e.g. a webcam)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # atau nilai lain yang lebih rendah
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # atau nilai lain yang lebih rendah


# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLOv10 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Convert BGR to RGB for displaying in matplotlib
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display the annotated frame using matplotlib
    plt.imshow(annotated_frame_rgb)
    plt.pause(0.001)  # Pause to update the plot

    # Exit the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release the video capture object
cap.release()
plt.close()