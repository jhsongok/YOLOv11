import os
import cv2
import requests
import supervision as sv
from ultralytics import YOLO

# `coco.names` 파일 다운로드
def download_coco_names():
    url = "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true"
    file_path = os.path.join(os.getcwd(), 'coco.names')
    
    if not os.path.exists(file_path):
        response = requests.get(url)
        with open(file_path, 'w') as file:
            file.write(response.text)
        print(f"coco.names downloaded and saved to {file_path}")
    else:
        print(f"coco.names already exists at {file_path}")

download_coco_names()

# `coco.names` 파일을 읽어 클래스 이름을 설정
def load_class_names(file_path):
    with open(file_path, 'r') as file:
        class_names = file.read().strip().split('\n')
    return class_names

class_names = load_class_names('coco.names')

model = YOLO('yolo11n.pt')

corner_annotator = sv.BoxCornerAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture('sample_video.mp4')
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 0: 상하대칭, 1 : 좌우대칭 

    results = model(frame)

    detections = sv.Detections.from_ultralytics(results[0])

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections['class_name'], detections.confidence)
    ]
    
    print(labels)

    # Bounding box에 대한 어노테이션
    annotated_image = corner_annotator.annotate(scene=frame, detections=detections)

    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    cv2.imshow('result', annotated_image)
    
    if cv2.waitKey(1) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()