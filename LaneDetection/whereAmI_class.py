import cv2
import numpy as np
import time

VIDEO_PATH = r"C:\Users\User\Desktop\성민이 깃헙\simulation-autonomousDriving\LaneDetection\no_gps_obstacle#13.avi"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("="*20)
    print("Camera Open Failed")
    print("="*20)
    exit()

_, frame = cap.read()

class WhereAmI:
    def __init__(self):
        # pixel white, yellow
        self.lower_white = np.array([0, 0, 200], dtype=np.uint8) # 0 ~ 2^8 - 1, 0~255
        self.upper_white = np.array([180, 30, 255], dtype=np.uint8)

        self.lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        self.upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
        
        # putText argument
        self.position = (640, 320)  
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_color = (0, 0, 255)  
        self.line_type = 3
        
        # roi 
        self.x1, self.y1, self.width1, self.height1 = int(frame.shape[1] * 0.20), int(frame.shape[0] * 0.8), 300, 200 # 왼쪽
        self.x2, self.y2, self.width2, self.height2 = int(frame.shape[1] * 0.55), int(frame.shape[0] * 0.8), 300, 200   # 오른쪽
        self.thresh_center_x3, self.thresh_center_y3, self.width3, self.height3 = int(frame.shape[1] * 0.15 + 400), int(frame.shape[0] - 150), 50, 150 # 가운데 
        self.crosswalk_center_x4, self.crosswalk_center_y4, self.width4, self.height4 = 0, int(frame.shape[0] * 0.7), int(frame.shape[1]), int(frame.shape[0] - frame.shape[0] * 0.7) # 횡단보도
        
        # 1차선일 때, 중앙선 침법 threshold
        self.crossing_centerline_thres = 10
        
        # default, start_time은 횡단보도용
        self.lane_info = "LANE 3" 
        self.temp_lane_info = ""
        self.start_time = time.time()
        self.flag = False

    # 동영상 저장
    def video_record(self):
        output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        record_method = cv2.VideoWriter(f'output_GPS13#_NoObstacle-case2.mp4', fourcc, fps, (output_width, output_height))
        return record_method
    
    # 횡단보도 검출, output은 윤곽
    def find_contour(self, roi_img, min_area = 1500):
        gray = cv2.cvtColor(roi_img.copy(), cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.bilateralFilter(gray, d=3, sigmaColor=10, sigmaSpace=5) # noise를 제거하되, edge부분을 지켜주는 필터
        thresh = cv2.threshold(blur_frame, 200, 255, cv2.THRESH_BINARY)[1] # 200~255 : 흰색, 200 미만 : 검정색
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crosswalk_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) == 4 and min_area < area: 
                crosswalk_contours.append(contour)

        return crosswalk_contours           

    # 해당 roi imgs
    def roi_imgs(self, frame):
        roi1 = frame[self.y1:self.y1+self.height1, self.x1:self.x1+self.width1]
        roi2 = frame[self.y2:self.y2+self.height2, self.x2:self.x2+self.width2]
        roi3 = frame[self.thresh_center_y3:self.thresh_center_y3+self.height3, self.thresh_center_x3:self.thresh_center_x3+self.width3]
        roi4 = frame[self.crosswalk_center_y4:self.crosswalk_center_y4+self.height4, self.crosswalk_center_x4:self.crosswalk_center_x4+self.width4]
        
        return roi1, roi2, roi3, roi4   
    
    # == white, yellow, above pixel들 개수 세기 == 
    def count_white_pixel(self, roi_frame):
        white_pixel_num = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV), self.lower_white, self.upper_white))
        return white_pixel_num
    
    def count_yellow_pixel(self, roi_frame):
        yellow_pixel_num = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV), self.lower_yellow, self.upper_yellow))
        return yellow_pixel_num
    
    def count_above_yellow_pixel(self, roi_frame):
        above_yellow_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV), self.lower_yellow, self.upper_yellow))
        return above_yellow_pixel
    
    # 확인차 roi img들에 rectangle 그리는 것
    def draw_rectangle(self, frame):
        cv2.rectangle(frame, (self.x1, self.y1), (self.x1+self.width1, self.y1+self.height1), (0, 255, 0), 2)
        cv2.rectangle(frame, (self.x2, self.y2), (self.x2+self.width2, self.y2+self.height2), (0, 255, 0), 2)
        cv2.rectangle(frame, (self.thresh_center_x3, self.thresh_center_y3), (self.thresh_center_x3+self.width3, self.thresh_center_y3+self.height3), (0, 255, 0), 2)
        cv2.rectangle(frame, (self.crosswalk_center_x4, self.crosswalk_center_y4), (self.crosswalk_center_x4+self.width4, self.crosswalk_center_y4+self.height4), (0, 255, 0), 2)  

        return frame

    # 몇 차선인지 검출
    def detect(self, frame):
        roi1, roi2, roi3, roi4 = self.roi_imgs(frame)
        
        left_white_pixel = self.count_white_pixel(roi1)
        left_yellow_pixel = self.count_yellow_pixel(roi1)
        right_white_pixel = self.count_white_pixel(roi2)
        right_yellow_pixel = self.count_yellow_pixel(roi2)
        
        self.draw_rectangle(frame)
        
        crosswalk_exist = self.find_contour(roi4)
        
        # 차선 : 노, 흰, 흰, 노
        if self.flag == False:
            if self.lane_info == "LANE 1" or self.lane_info == "Encroaching The CenterLine":
                above_pixel_cnts = self.count_above_yellow_pixel(roi3)

            # 횡단보도가 아닐 시
            if len(crosswalk_exist) > 0 and len(crosswalk_exist) < 10:
                if self.lane_info =="LANE 3":
                    if right_white_pixel > right_yellow_pixel:
                        self.lane_info = "LANE 2"

                elif self.lane_info == "LANE 2":
                    if left_white_pixel < left_yellow_pixel:
                        self.lane_info = "LANE 1"
                    elif right_white_pixel < right_yellow_pixel:
                        self.lane_info = "LANE 3"
                        
                elif self.lane_info == "LANE 1":
                    if left_white_pixel > left_yellow_pixel:
                        self.lane_info = "LANE 2"
                    elif above_pixel_cnts >= self.crossing_centerline_thres:
                        self.lane_info = "Possible The Crossing CenterLine"
                
                # heading이 왼쪽으로 꺾이면, 중앙선을 바라보는 것으로 노란색 검출
                elif self.lane_info == "Possible The Crossing CenterLine":
                    if right_white_pixel > right_yellow_pixel:
                        self.lane_info = "LANE 1" # LANE 1
                    elif right_white_pixel < right_yellow_pixel:
                        self.lane_info = "Encroaching The CenterLine"
                
                # 역주행 차선들 기점으로, 어차피 중앙선을 넘으면 실격이니 3차선에 있다고만 가정
                elif self.lane_info == "Encroaching The CenterLine":
                    if above_pixel_cnts >= self.crossing_centerline_thres:
                        self.lane_info = "Possible The Crossing CenterLine"
                    elif right_white_pixel > right_yellow_pixel:
                        self.lane_info = "LANE 1"
                
                # 이는 하드코딩을 위한 것으로, 횡단보도를 계속 유지하고 있다는 것을 유지(차선이 점선)
                elif self.lane_info == "Crossing The Pedestrian Crosswalk":
                    if len(crosswalk_exist) >= 2 and len(crosswalk_exist) < 4: # 이것은 횡단보도를 건너고, 차선을 달릴 때를 말함
                        self.lane_info = self.temp_lane_info # 횡단보도 이전의 값을 받아옴
            
            # 횡단보도 검출, 바로 위의 코드보다 밑에 있는 것이 먼저 실행됌
            elif len(crosswalk_exist) >= 10:
                # self.lane_info가 횡단보도이기전 값을 temp에 저장
                if self.lane_info != "Crossing The Pedestrian Crosswalk":
                    self.temp_lane_info = self.lane_info # 횡단보도 건너기 전의 값을 받아옴(이전의 값을 최대한 유지한다고 가정)
                self.lane_info = "Crossing The Pedestrian Crosswalk"
                self.start_time = time.time()
                self.flag = True
        
        # 횡단보도이면 하드코딩으로 1초간 유지
        elif self.flag:
            self.lane_info = "Crossing The Pedestrian Crosswalk"
            if time.time() - self.start_time >= 1:
                self.flag = False
        
        return frame, self.lane_info
    
    def run(self):
        record = self.video_record()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("="*20)
                print("No Frame")
                print("="*20)
                break
            
            update_frame, _ = self.detect(frame)
            cv2.putText(update_frame, self.lane_info, self.position, self.font, self.font_scale, self.font_color, self.line_type)
            record.write(frame)
            cv2.imshow("Result", frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print("=" * 20)
                print("KeyBoard Interrupt")
                print("=" * 20)
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
def main():
    hee = WhereAmI()
    hee.run()
    
if __name__ == "__main__":
    main()
