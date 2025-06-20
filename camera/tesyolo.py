import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('D:/RPSLO object detection/Robot-Pembersih-Sampah-Laut-Otonom/ModelYolo/yolov8n.pt')

# Open the video capture device (e.g. a webcam)
cap = cv2.VideoCapture(0)

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()