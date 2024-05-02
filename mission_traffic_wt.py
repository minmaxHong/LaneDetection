#!/usr/bin/env python3
# -- coding: utf-8 --
import rospy
import torch
import cv2
import numpy as np
import time
import os, sys
from std_msgs.msg import String, Bool
from macaron_6_svac.msg import Traffic, obj_info
from morai_msgs.msg import GPSMessage
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
# from TrafficLightDetector import TrafficLightDetector
import torchvision.transforms as transforms
from std_msgs.msg import Float32


# model.to(DEVICE)

class MissionTraffic:
    def __init__(self):
        self.bridge = CvBridge()
        self.img_flag = False
        self.go_cnt_5 = 0
        self.go_cnt_8 = 0
        self.close_crosswalk_cnt = 0
        self.close_crosswalk_cnt_thres = 2
        self.far_crosswalk_cnt = 0
        self.far_crosswalk_cnt_thres = 7
        self.trafficlight_info = None
        self.trafficlight_info_nodetect = 0
        self.trafficlight_info_bool = False
        self.trafficlight_info_nodetect_thres = 10
        self.path_info = None
        self.t = 0
        self.s = 0
        self.stop_s = [[6960.3-5, 7000.9-5],
                       [6960.3-5, 7000.9-5]]

        self.pub_stop = rospy.Publisher('/stop', Bool, queue_size=1)

        WEIGHT_PATH = '/home/takrop/catkin_ws/src/macaron_6_svac/pt/0nly_traffic.pt'
        self.model = YOLO(WEIGHT_PATH)
        self.model.to('cuda')
        
        # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def process_img(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.bilateralFilter(gray, d=3, sigmaColor=10, sigmaSpace=5)
        thresh = cv2.threshold(blur_frame, 200, 255, cv2.THRESH_BINARY)[1]
        return thresh

    def roi(self, frame, thresh, distance=True):
        close_mask = np.zeros_like(thresh)
        ignore_mask = 255
        height, width = frame.shape[:2]
        
        if distance:
            close_bottom_left = [0, height]
            close_top_left = [0, height * 0.6]
            close_bottom_right = [width, height]
            close_top_right = [width, height * 0.6]
            
            close_vertice = np.array([[close_bottom_left, close_top_left,
                                    close_top_right, close_bottom_right]], dtype=np.int32)
            
            cv2.fillPoly(close_mask, close_vertice, ignore_mask)
            close_masked_img = cv2.bitwise_and(thresh, close_mask)
            
            return close_masked_img, close_vertice

    def find_contour(self, roi_img, min_area=1300):
        contours, _ = cv2.findContours(roi_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crosswalk_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) == 4 and min_area < area:
                crosswalk_contours.append(contour)
                
        return crosswalk_contours    
       
    def close_detect_crosswalk(self):
        frame = self.img
        thresh = self.process_img(frame)
        region_interest, close_vertice = self.roi(frame, thresh, distance=True)
        crosswalk_contours = self.find_contour(region_interest)
        
        cv2.putText(frame, "Region of Close Crosswalk", close_vertice[0][1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(frame, close_vertice[0][0], close_vertice[0][1], (255, 0, 0), 2)
        cv2.line(frame, close_vertice[0][1], close_vertice[0][2], (255, 0, 0), 2)
        cv2.line(frame, close_vertice[0][2], close_vertice[0][3], (255, 0, 0), 2)
        cv2.line(frame, close_vertice[0][3], close_vertice[0][0], (255, 0, 0), 2)

        cv2.drawContours(frame, crosswalk_contours, 0, (0, 255, 0), 2)
        if len(crosswalk_contours) > 4:
            return True
        else:
            return False
    
    def interpolated(self, frame):
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        return morph

    def persp_transform(self, frame):
        height, width = frame.shape[:2]
        
        # 720x1280x3 일때
        original_bottom_left = [460, 460]
        original_top_left = [530, 400]
        original_bottom_right = [840, 460]
        original_top_right = [770, 400]
        

        # 480x640x3 일때
        scale_width = 640 / 1280
        scale_height = 480 / 720

        bottom_left = [int(original_bottom_left[0] * scale_width), int(original_bottom_left[1] * scale_height)]
        top_left = [int(original_top_left[0] * scale_width), int(original_top_left[1] * scale_height)]
        bottom_right = [int(original_bottom_right[0] * scale_width), int(original_bottom_right[1] * scale_height)]
        top_right = [int(original_top_right[0] * scale_width), int(original_top_right[1] * scale_height)]

        # # bird eye view 선 그리기
        # cv2.line(frame, tuple(bottom_left), tuple(top_left), (0, 255, 0), 3)
        # cv2.line(frame, tuple(bottom_left), tuple(bottom_right), (0, 255, 0), 3)
        # cv2.line(frame, tuple(bottom_right), tuple(top_right), (0, 255, 0), 3)
        # cv2.line(frame, tuple(top_right), tuple(top_left), (0, 255, 0), 3)

        src_points = np.float32([bottom_left, top_left, bottom_right, top_right]) # 좌/하단, 좌/상단, 우/하단, 우/상단
        dst_points = np.float32([[0, height], [0, 0], [width, height], [width, 0]])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 원근 변환 적용
        warped = cv2.warpPerspective(frame, matrix, (width, height))
        return warped

    def far_detect_crosswalk(self):
        frame = self.img
        processed_frame = self.process_img(frame)
        persp_frame = self.persp_transform(processed_frame)
        interpolated_frame = self.interpolated(persp_frame)
        far_crosswalks_contours = self.find_contour(interpolated_frame)
        
        print(len(far_crosswalks_contours))
        if len(far_crosswalks_contours) > 3:
            return True
        else:
            return False

    def traffic_detect(self):
        label = ''
        start_time = time.time()
        # ROI 설정
        img = self.img
        img = cv2.resize(img, (640,480))

        self.img = img
        input = img.copy()
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = input.astype(np.float32)
        input /=255.0
        input = torch.tensor(input).permute(2,0,1)
        input = torch.unsqueeze(input, dim = 0)
        self.model.to('cuda')
        results = self.model(input, verbose = False)
        boxes = results[0].boxes.cpu().numpy().data
        index = 3
        max_list = []
        if len(boxes) != 0:
            max_tuple = boxes[0]
            max_value = boxes[0][index]
        # 각 튜플의 지정된 인덱스 값을 비교하여 가장 작은 값을 가진 튜플을 찾습니다.
            for element in boxes:
                current_value = element[index]
                if current_value > max_value:
                    max_value = current_value
                    max_tuple = element
            max_list.append(max_tuple)
    
        #print(max_list)
        # boxes[:,3]
        for box in max_list:
            if int(box[3]) <= 240:
                left = int(box[0])
                top = int(box[1]) 
                right = int(box[2])
                bottom = int(box[3])
                label_index = int(box[5])
                label = results[0].names[label_index]
                self.trafficlight_info = label
                cv2.putText(img, label, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.rectangle(img, (left, bottom), (right, top), (0, 255, 0), 2)
                return label
    
    # 음영구간
    def run_5(self, img):
        self.img = img
        self.trafficlight_info = self.traffic_detect()

        # 신호등 검출의 유/무
        if self.trafficlight_info == None: # 무
            self.trafficlight_info_bool = False
        else: 
            self.trafficlight_info_bool = True # 유

        # 멀리있는 횡단보도 개수 / 가까이 있는 횡단보도 개수
        if self.far_detect_crosswalk():
            self.far_crosswalk_cnt += 1
        if self.close_detect_crosswalk():
            self.close_crosswalk_cnt += 1

        # 신호등 검출할 시
        if self.trafficlight_info_bool:

            # case 1: 멀리있는 횡단보도만 보이고, 가까이 있는 것은 안보일 때 일단 정지선 앞까지는 가야함 -> 'go'
            # case 2: 가까이 있는 횡단보도만 보일때 -> 'stop'
            if self.trafficlight_info == 'Stop':
                # 멀리 있는 횡단보도가 임계점 넘으면 stop
                if self.close_crosswalk_cnt >= 1:
                    self.path_info = "stop"

                elif self.far_crosswalk_cnt >= self.far_crosswalk_cnt_thres:
                    self.path_info = "stop"
                
                elif self.far_crosswalk_cnt < self.far_crosswalk_cnt_thres:
                    self.path_info = "go"

            # 'Go'일 시
            # 무조건 가야함
            elif self.trafficlight_info == 'Go':
                self.path_info = "go"
                self.go_cnt_5 += 1
                if self.go_cnt_5 >= 5:
                    self.far_crosswalk_cnt = 0
                    self.close_crosswalk_cnt = 0

        if self.path_info == "stop":
            self.pub_stop.publish(True)

        # cv2.imshow("detect_image",self.img)
        
        # print("================================================")
        # print(f'Close Crosswalk : {self.close_crosswalk_cnt}')
        # print(f'Far Crosswalk {self.far_crosswalk_cnt}')
        # print(f'PATH INFO : {self.path_info}')
        # print("================================================\n\n")

        #print('DEVICE', DEVICE)
    
    # 음영구간 아님
    def run_8_10(self, img):
        self.img = img
        self.trafficlight_info = self.traffic_detect()
        
        # 신호등 검출의 유/무
        if self.trafficlight_info == 'Stop':
            self.pub_stop.publish(True)

        # cv2.imshow("detect_image",self.img)
        # print("================================================")
        # print(f'Close Crosswalk : {self.close_crosswalk_cnt}')
        # print(f'Far Crosswalk {self.far_crosswalk_cnt}')
        # print(f'PATH INFO : {self.path_info}')
        # print("================================================\n\n")

    def update_img(self, img):
        self.img = img
        
def main():
    rospy.init_node("object_detection")
    traffic = MissionTraffic()
    
    while not rospy.is_shutdown():
        if traffic.img_flag:
            # GPS 음영구간
            # -- 성민 --
            # if 670 <= traffic.s <= 720: # 원래 700~750
            traffic.run_5()
            # # 2번째 신호등 구간
            # elif 885 < traffic.s < 915: # 915~945
            #     traffic.run_8_10()
            # # 3번째 구간
            # elif 995< traffic.s < 1017: # 1015~1047
            #     traffic.run_8_10()
            
            traffic.img_flag = False
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()