import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (replace 'best.pt' with your trained model file)
model = YOLO('F:\sih\best.pt')  # Ensure that you provide the correct path to your trained YOLOv8 model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process the video stream frame by frame
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLOv8 detection on the frame
    results = model(frame)

    # Extract bounding boxes and labels from the results
    for r in results:
        boxes = r.boxes  # Extract bounding boxes
        for box in boxes:
            # Extract the bounding box coordinates and confidence score
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = model.names[cls]  # Class label

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("YOLOv8 Crack Detection", frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

