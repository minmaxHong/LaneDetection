import cv2
import torch
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import bird_eye_view

model_path = r"C:\Users\User\Desktop\ulta\tusimple_18.pth"
video_path = r"C:\Users\User\Desktop\ulta\음영구간_1.avi"
# output_video_path = r"C:\Users\H_\Desktop\ulta\detected_lanes.mp4"  
model_type = ModelType.TUSIMPLE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_gpu = False
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

cap = cv2.VideoCapture(video_path)

# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    output_frame = lane_detector.detect_lanes(frame)
    bed_frame = bird_eye_view.persp_transform(output_frame)
    
    # out.write(output_frame)

    cv2.imshow("Detected lanes", output_frame)
    # cv2.imshow('BED', bed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
