import cv2
from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
model = YOLO("trafficlight_v1.pt")

video_path = 0

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes.data:
            x1,y1,x2,y2,conf,cls = box.tolist()
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', 
                        (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0, 255, 0), 2)
        cv2.imshow("Video",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()