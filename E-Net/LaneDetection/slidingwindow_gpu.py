import numpy as np
import cv2
import sys
import datetime
from matplotlib import pyplot as plt
import time
import torch
import torchvision.transforms as transforms

class LaneDetection:
    def __init__(self, video_path, window_search=9, color=(0, 255, 0), thickness=3, frame_cnt=0):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.vertices = None
        self.window_search = window_search
        self.color = color
        self.thickness = thickness
        self.frame_cnt = frame_cnt

        # ======== GPU(CUDA) 써봄 ===========
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # cuda사용가능

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])
        # ===================================
    # == Img Processing == 
    def img_processing(self, frame):
        # NumPy 배열을 PyTorch 텐서로 변환
        tensor_frame = self.transform(frame).to(self.device)

        # PyTorch로 HSV 변환
        hsv_frame = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), device=self.device)

        # # 색상 선택을 위한 임계값 정의
        # white_lower_thres = torch.tensor([0, 0, 200], device=self.device)
        # white_upper_thres = torch.tensor([255, 30, 255], device=self.device)
        # yellow_lower_thres = torch.tensor([20, 100, 100], device=self.device)
        # yellow_upper_thres = torch.tensor([30, 255, 255], device=self.device)

        # # 마스크 생성
        # white_mask = torch.tensor(cv2.inRange(hsv_frame, white_lower_thres, white_upper_thres), device=self.device)
        # yellow_mask = torch.tensor(cv2.inRange(hsv_frame, yellow_lower_thres, yellow_upper_thres), device=self.device)
        # mask = torch.bitwise_or(white_mask, yellow_mask)

        # # 마스크를 이용하여 이미지를 필터링
        # masked_img = torch.bitwise_and(tensor_frame, tensor_frame, mask=mask)

        # # Gray scale로 변환
        # gray_img = torch.mean(masked_img, dim=0).byte()

        return hsv_frame
           
    # == Region Of Interest == 
    def roi(self, frame):
        mask = np.zeros_like(frame)
        
        if len(frame.shape) > 2:
            channel_cnt = frame.shape[2]
            ignore_mask_color = (255, ) * channel_cnt
            
        else:
            ignore_mask_color = 255
        
        # Bottom left, Top left, Top right, Bottom right
        height, width = frame.shape[:2]
        bottom_left = [width * 0.1, height * 0.95]
        top_left = [width * 0.4, height * 0.6]
        bottom_right = [width * 0.9, height * 0.95]
        top_right = [width * 0.6, height * 0.6]
        
        self.vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32) 
        
        cv2.fillPoly(mask, self.vertices, ignore_mask_color)
        masked_img = cv2.bitwise_and(frame, mask)
        
        return masked_img        
    
    # == Bird-eye-view == 
    def bed(self, frame):
        height, width = frame.shape[:2]  # y, x
        
        bottom_left, top_left, top_right, bottom_right = np.float32(self.vertices[0])

        src_points = np.float32([bottom_left, top_left, top_right, bottom_right])
        dst_points = np.float32([[0, height], [0, 0], [width, 0], [width, height]])
        perspective_mat = cv2.getPerspectiveTransform(src_points, dst_points)
        inverse_mat = cv2.getPerspectiveTransform(dst_points, src_points)

        bed_frame = cv2.warpPerspective(frame, perspective_mat, (width, height))
        
        return bed_frame, inverse_mat
    
    # == Left Lines & Right Lines == 
    def track_lanes_initialize(self, frame):
        histogram = np.sum(frame[int(frame.shape[0] / 2):, :], axis=0)     
        
        out_img = np.dstack((frame, frame, frame)) * 255
        
        # ========= width =============
        midpoint = int(histogram.shape[0] / 2)
        left_width_base = np.argmax(histogram[:midpoint]) 
        right_width_base = np.argmax(histogram[midpoint:]) + midpoint
        # =============================
        
        window_height = int(frame.shape[0] / self.window_search)
        
        # ====== frame에서 lane인 것들의 index값들을 가져옴 ======
        nonzero = frame.nonzero()
        nonzero_height = np.array(nonzero[0])
        nonzero_width = np.array(nonzero[1])
        # =======================================================
        
        left_width_current = left_width_base
        right_width_current = right_width_base
        
        margin = 100
        minpix = 50
        
        left_lane_inds = []
        right_lane_inds = []
        
        for sub_window in range(self.window_search):
            # ================= sub_window 경계 계산 ===================
            sub_window_height_lower = int(frame.shape[0] - (sub_window + 1) * window_height)
            sub_window_height_higher = int(frame.shape[0] - sub_window * window_height)
            
            sub_window_width_left_lower = left_width_current - margin
            sub_window_width_left_higher = left_width_current + margin
            sub_window_width_right_lower = right_width_current - margin
            sub_window_width_right_higher = right_width_current + margin
            # ==========================================================
            
            cv2.rectangle(out_img, (sub_window_width_left_lower, sub_window_height_lower),
                                (sub_window_width_left_higher, sub_window_height_higher),
                               self.color, self.thickness)    
            cv2.rectangle(out_img, (sub_window_width_right_lower, sub_window_height_lower),
                                (sub_window_width_right_higher, sub_window_height_higher), 
                                self.color, self.thickness)
            
            # 2D -> 1D로 바뀌값들임, 왼쪽/오른쪽 차선 기준으로의 sub_window에 있는 차선의 index값들
            detect_left_inds = ((nonzero_height >= sub_window_height_lower) &
                              (nonzero_height < sub_window_height_higher) &
                              (nonzero_width >= sub_window_width_left_lower) &
                              (nonzero_width < sub_window_width_left_higher)).nonzero()[0]
            
            detect_right_inds = ((nonzero_height >= sub_window_height_lower) &
                               (nonzero_height < sub_window_height_higher) &
                               (nonzero_width >= sub_window_width_right_lower) &
                               (nonzero_width < sub_window_width_right_higher)).nonzero()[0]
            
            left_lane_inds.append(detect_left_inds)
            right_lane_inds.append(detect_right_inds)
            
            if len(detect_left_inds) > minpix:
                left_width_current = int(np.mean(nonzero_width[detect_left_inds]))
            
            if len(detect_right_inds) > minpix:
                right_width_current = int(np.mean(nonzero_width[detect_right_inds]))
            
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        left_width = nonzero_width[left_lane_inds]
        left_height = nonzero_height[left_lane_inds]
        right_width = nonzero_width[right_lane_inds]
        right_height = nonzero_height[right_lane_inds]
        
        # 곡률값들 반환해서 track_lanes_update로
        left_fit = np.polyfit(left_height, left_width, 2)
        right_fit = np.polyfit(right_height, right_width, 2)

        return left_fit, right_fit
    
    def track_lanes_update(self, frame, left_fit, right_fit):
        nonzero = frame.nonzero()
        nonzero_height = np.array(nonzero[0])
        nonzero_width = np.array(nonzero[1])

        margin = 100

        left_lane_inds = ((nonzero_width > (left_fit[0]*(nonzero_height**2) + left_fit[1]*nonzero_height + left_fit[2] - margin)) & (nonzero_width < (left_fit[0]*(nonzero_height**2) + left_fit[1]*nonzero_height + left_fit[2] + margin))) 
        right_lane_inds = ((nonzero_width > (right_fit[0]*(nonzero_height**2) + right_fit[1]*nonzero_height + right_fit[2] - margin)) & (nonzero_width < (right_fit[0]*(nonzero_height**2) + right_fit[1]*nonzero_height + right_fit[2] + margin)))  

        leftx = nonzero_width[left_lane_inds]
        lefty = nonzero_height[left_lane_inds] 
        rightx = nonzero_width[right_lane_inds]
        righty = nonzero_height[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit ,right_fit
    
    # 곡률(쌍곡선들 계산) -> 오른쪽, 왼쪽 차선들의 곡률 구하는 식을 lane_fill_poly에서 씀
    def get_val(self, y, poly_coeff):
        return poly_coeff[0] * y ** 2 + poly_coeff[1] * y + poly_coeff[2]
    
    def lane_fill_poly(self, bed_frame, undist, left_fit, right_fit, inverse_mat):

        ploty = np.linspace(0, bed_frame.shape[0]-1, bed_frame.shape[0])
        left_fitx = self.get_val(ploty, left_fit)
        right_fitx = self.get_val(ploty, right_fit)
        
        warp_zero = np.zeros_like(bed_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = cv2.warpPerspective(color_warp, inverse_mat, (bed_frame.shape[1], bed_frame.shape[0])) 

        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
            
        return result

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                print('==========================')
                print('No Frame')
                print('==========================')
                sys.exit()
            
            processed_frame = self.img_processing(frame)
            processed_frame = processed_frame.cpu().numpy()
            # roi_frame = self.roi(processed_frame)
            # bed_frame, inverse_mat = self.bed(roi_frame)
            # left_fit_1, right_fit_1 = self.track_lanes_initialize(bed_frame)
            # left_fit, right_fit = self.track_lanes_update(bed_frame, left_fit_1, right_fit_1)
            # colored_lane = self.lane_fill_poly(bed_frame, frame, left_fit, right_fit, inverse_mat)
            
            # colored_lane = cv2.cuda_GpuMat()
            
            cv2.imshow('Result', processed_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = "solidWhiteRight.mp4"
    lane_detection_system = LaneDetection(video_path)
    lane_detection_system.run()
    
if __name__ == "__main__":
    main()
