import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

class Last_Stop:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if self.cap is None:
            print('==========================')
            print('Cap Warning')
            print('==========================')
            sys.exit()
    
    def img_processing(self, frame):
        # Noise Remove
        blur_frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # White        
        lower_white = np.array([200, 200, 200])  # 하얀색의 최소값
        upper_white = np.array([255, 255, 255])  # 하얀색의 최대값
        
        white_amsk
        masked_img = cv2.bitwise_and(frame, frame, mask = )
        
        return blur_frame
    
    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                print('==========================')
                print('No Frame')
                print('==========================')
                break  # 비디오 재생이 끝날 때 루프 종료
                        
            processed_frame = self.img_processing(frame)
            concat_frame = np.hstack((processed_frame, frame))

            cv2.imshow('Video', concat_frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
        # 비디오 재생이 끝난 후에 실행되어야 할 부분
        self.cap.release()
        cv2.destroyAllWindows()
            

def main():
    video_path = 'C:\\Users\\User\\Desktop\\차선검출코드\\Data\\video\\morai_test_video.mp4'
    stop_lane_system = Last_Stop(video_path)
    stop_lane_system.run()


if __name__ == "__main__":
    main()
