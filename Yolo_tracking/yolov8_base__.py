import cv2
from ultralytics import YOLO

WEIGHT_PATH = r"C:\Users\User\Desktop\성민이 깃헙\Pt\04_29.pt"
video_path = r"C:\Users\User\Desktop\성민이 깃헙\LaneDetection\Data\carbackhead3.avi"
output_video_path = r"output_video.mp4"  # Specify the path for the output video file
model = YOLO(WEIGHT_PATH)
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_fps = cap.get(cv2.CAP_PROP_FPS)
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (output_width, output_height))

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)  # Write the frame to the output video
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
