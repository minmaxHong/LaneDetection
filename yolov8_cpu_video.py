import cv2
import sys
from ultralytics import YOLO

class object_detection:
    def __init__(self, weight_path=None, video_path=None, conf=0.5, target_size=(640, 480)):
        self.video_path = video_path
        self.weight_path = weight_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.model = YOLO(self.weight_path)
        self.conf = conf
        self.target_size = target_size
        
    def run(self):
        while True:
            ret, frame = self.cap.read()
            
            if ret is None:
                print("====================")
                print("==== No Frame !! ===")
                print("====================")
                sys.exit()
            
            result = self.model(frame, conf=self.conf)
            
            resized_frame = cv2.resize(result[0].plot(), self.target_size)
            cv2.imshow('Detect Video', resized_frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
        cv2.destroyAllWindows()
        self.cap.release()
                
    
def main():
    video_path = "project_video.mp4"
    object_detection_system = object_detection(weight_path="yolov8n.pt", video_path=video_path)
    object_detection_system.run()
    
if __name__ == "__main__":
    main()
