import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

WEIGHT_PATH = r"C:\Users\User\Desktop\성민이 깃헙\Pt\04_29.pt"
VIDEO_PATH = r"C:\Users\User\Desktop\성민이 깃헙\LaneDetection\Data\carbackhead2.avi"

model = YOLO(WEIGHT_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

center_point = (640, -30) # width, height 
pixel_per_meter = 100

txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True, verbose = False)
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            annotator.box_label(box, label=str(track_id), color=bbox_clr)
            
            # bbox : left, bottom, right, top
            x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)  # width, height
            center_point = (x1, im0.shape[0]) # width, height
            annotator.visioneye(box, center_point)
            distance = (math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2))/pixel_per_meter
            
            text_size, _ = cv2.getTextSize(f"Distance: {distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX,1.2, 3)
            cv2.rectangle(im0, (x1, y1 - text_size[1] - 10),(x1 + text_size[0] + 10, y1), txt_background, -1)
            cv2.putText(im0, f"Distance: {distance:.2f} m",(x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2,txt_color, 3)

    cv2.imshow("visioneye-distance-calculation", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()