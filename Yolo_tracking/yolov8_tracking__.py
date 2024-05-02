import cv2
import torch
import numpy as np
from ultralytics import YOLO
# boxes : bbox 좌표 , tensor로 반환, torch.Size는 행이 추가되면 객체 늘어남, 열이면 bbox정보(x, y, w, h)
# keypoints : 객체 내 주요 지점을 감지한 경우
# masks : 객체의 분할된 mask 정보
# names : {0: 'Traffic Cone', 1: 'Person', 2: 'Vehicle'}
# obb : 객체를 감싸는 회전된 bbox 정보
# orig_img : rgb img, dtype : uint8
# orig_shape : 원본 img 크기
# path : 원본 이미지의 파일 경로
# probs : 각 객체 클래스에 대한 확률
# save_dir : 결과를 저장할 디렉토리 경로
# speed : 객체 감지 작업의 속도 정보 (전처리, 추론, 후처리)
# boxes.id.int().cpu() : 객체의 고유 식별 번호, 반환 값 tensor

# Load the YOLOv8 model
WEIGHT_PATH = r"C:\Users\User\Desktop\성민이 깃헙\Pt\04_29.pt"
video_path = r"C:\Users\User\Desktop\성민이 깃헙\LaneDetection\Data\carbackhead3.avi"
output_video_path = r"output_video.mp4"  
model = YOLO(WEIGHT_PATH)
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # persist : frame간에 추적을 지속한다 (True)
        # results는 항상 results[0]값에 값을 가짐
        img1 = cv2.resize(frame, dsize = (640, 480))
        results = model.track(img1, persist=True, verbose = False)
        # print(results[0].names) # class {0: 'Traffic Cone', 1: 'Person', 2: 'Vehicle'}
        boxes = results[0].boxes.cpu().numpy().data
        # print(f'Track Ids : {track_ids}')
        # print(f'Boxes Info : {boxes} {boxes[0]}') # (left, bottom, right, top, id고유번호, conf, class인덱스)
        if len(boxes) > 0:
            for box in boxes:
                print(box)
                left, top, right, bottom, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[5] # box[5]는 conf가 소수점이라 int casting해주면 안됩니다.
                label_name = results[0].names[int(box[6])] 
                print(label_name)
                print(conf)
                
                if conf > 0.5:
                    cv2.putText(img1, label_name, (left - 20, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(img1, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow('Frame', img1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()