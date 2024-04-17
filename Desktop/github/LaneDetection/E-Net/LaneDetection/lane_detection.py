#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocess_cv2__ import *
# import rospy
import sys, os
import cv2
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.lanenet.LaneNet import LaneNet


currentPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
root='./test_dataset/test_video.mp4'

class LaneDetection():
    def __init__(self, webcam_port=3):
        global currentPath
        self.cap = cv2.VideoCapture(webcam_port)
        self.cap_video = cv2.VideoCapture(root)
        self.now = datetime.now() # estimate time
        self.current = 0
        self.model_path = currentPath + '/log/ENet_best_150.pth' # 150으로 결정
        self.model = LaneNet(arch='ENet') #LaneNet class 생성

        # 네트워크 값 불러오기
        self.state_dict = torch.load(self.model_path, map_location='cuda:0') # map_location은 train한 cuda 번호랑 다를 수 있어서 넣어줌 (50은 아마 cuda:2로 학습)
        self.model.load_state_dict(self.state_dict) #model에 .pth 파일 불러오기
        self.lane = [[], []]
        for _ in range(180):
            self.lane[0].append(90)
            self.lane[1].append(270)

        self.currentBalanced = 0

        # plotting 
        self.y_center = []
        self.x_count = 0
        self.fromCenter = [0]
        self.detected = 0
     
    def getFrame(self): #frame 읽어오기
        ret, frame = self.cap.read()
        if ret:
            return frame
        
    def getFrame_video(self): #frame 읽어오기
        ret, frame = self.cap_video.read()
        #frame=cv2.rotate(frame,cv2.ROTATE_180)
        if ret:
            return frame

    def detect(self):
        current = time.time() # 현재 시간
    
        img = self.getFrame_video() #720x1280x3
        #img = self.getFrame() # 웹캠 
        #img = cv2.imread('/home/leejinho/catkin_ws/lane_corn.jpg', cv2.IMREAD_COLOR)
        img = cv2.resize(img, (320, 180)) #resize
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # BGR -> RGB로 변환인데 애초에 RGB로 되어있어서 굳이 사용할 필요 없을 듯
        #color, bordered_color, binary = getImages(img, self.model) #preprocess_lanenet에 있는 함수
        color, binary = getImages(img, self.model) #preprocess_lanenet에 있는 함수
        #float32형으로 바꿔준다
        dst = binary.astype(np.float32) #320x180
        dst = perspective_warp(dst, dst_size=(320, 180))
        cv2.imshow('bird_eye_view', dst)

        # inv = inv_perspective_warp(dst, dst_size=(320, 180)) 안쓰는데 굳이?
        out_img, curves, balanced, detect_true_false = sliding_window(dst) # return값으로 안 쓰는 것 모두 지움
        #out_img(sliding window), curves(차선에 대한 2차 함수), balanced(잘 모르겠음)만 사용함
        
        if detect_true_false: # 차선이 둘 다 검출이 됐다면
            self.lane = curves #최신 값으로 갱신
        else: # 둘 중에 하나라도 검출되지 않았다면
            print("Detect Fail!") # 검출 실패! -> 이전 값 계속 사용

        # color는 그냥 img, 왼쪽 차선이랑 오른쪽 차선의 곡선을 집어넣음
        img_, lane_figure = draw_lanes(color, self.lane[0], self.lane[1])
        img = cv2.addWeighted(img, 1, lane_figure, 0.7, 0) # 원래 이미지에 합성하는 과정

        fin_img = cv2.resize(img,(640,320))
        img__ = cv2.resize(img_, (640, 360))
        
        try:
            curverad = get_curve(img, curves[0], curves[1]) # img와 왼쪽 차선과 오른쪽 차선(2차 곡선)을 parameter 값으로 넣어줌
            centered, isOutliner = keepCenter(self.fromCenter, curverad[2])

            if isOutliner == 1:
                self.fromCenter.append(centered)
                self.y_center.append(self.fromCenter[-1])
                self.x_count += 1

            elif isOutliner == -1:
                self.y_center.append(self.fromCenter[-1])
                self.x_count += 1
            
            self.detected += 1

        except:
            self.y_center.append(self.fromCenter[-1])
            self.x_count += 1
        

        self.currentBalanced = balanced

        binary = cv2.resize(binary, (640, 360))
        img__ = cv2.cvtColor(img__, cv2.COLOR_BGR2RGB)
        dst = cv2.resize(dst, (640, 360))
        out_img = cv2.resize(out_img, (640, 360))
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        
        cv2.imshow("img__", fin_img) # 최종 이미지
    
        self.center = self.currentBalanced
  
        print("\nFrame : {}s\n\n\n".format(float(1 / (time.time() - current))))


if __name__ == "__main__":
    
    LD = LaneDetection(2) # default webcam port = 0 -> refer to definition
    
    #rospy.init_node("lane_detection")
    pre_time=time.time()    

    while True:
        # if time.time()-pre_time > 0.5:
            LD.detect()

            if cv2.waitKey(1) == ord('q'):
                break
            pre_time = time.time()

    # while not rospy.is_shutdown():
    #     if time.time()-pre_time>0.5:       
    #         ld_pub = rospy.Publisher('lane_center', Float64, queue_size=1)

    #         LD.detect()

    #         ld_pub.publish(LD.center)
    #         # time.sleep(0.2)
    #         if cv2.waitKey(1) == ord('q'):
    #             break
    #         pre_time=time.time()    
    
    print("Detected : {}%".format(LD.detected / LD.x_count * 100))
    print("Not detected : {}%".format(100 - LD.detected / LD.x_count * 100))
    
    LD.cap.release()
    cv2.destroyAllWindows()

    plt.scatter(range(LD.x_count), LD.y_center)
    plt.ylim([-0.2, 0.2])  
    plt.xlabel("frames")
    plt.ylabel("center")
    plt.title("Center Position")
    plt.savefig("./plots/{}_scatter.jpg".format(LD.now.strftime('%Y%m%d_%H%M%S%f')))
    plt.show()
    plt.close()