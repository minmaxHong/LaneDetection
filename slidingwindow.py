import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# ======== GPU(CUDA) 써봄 ===========
# opencv랑 호환성이 생각보다 떨어짐
# 이미지 변환을 해도 결국에는 다른 method에서 처리하려면 cpu로 변환해줘야함,,
# img_processing부터 hsv_frame을 gpu로 넣고 해봤는데 cpu보다 성능이 구렸음,, -> 이는 이유를 모르겠음
# ===================================

# 3/12 - 새벽
# class구현 다함 -> YOLO랑 class끼리 묶는건 시간문제 지금 당장도 할 수 있음
# GPU가 성능이 구릴리가 없는데, GPU로 img 처리했더니 성능이 좋지 않았음,,
# pytorch가 torch.transforms로 img전처리를 생각보다 빠르게 해주는데, 이를 이용해볼 예정인데 조금 걸릴듯
# 딥러닝 모델은 내 생각에는 "이진호"님 블로그 보니까 train에 따라서 성능이 매우 구린듯 -> 시간이 조금 남았으니 데이터셋을 따도 될지도..? -> 이는 3월20일부터 해볼 예정

# 3/12 - 학교
# pre-trained된 YOLO8 class짜가지고, LaneDetection이랑 Object Detection 합쳐보겠음.

class LaneDetection:
    def __init__(self, video_path, window_search=9, color=(0, 255, 0), thickness=3, frame_cnt=0):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.vertices = None
        self.window_search = window_search
        self.color = color
        self.thickness = thickness
        self.frame_cnt = frame_cnt

        

    # == Img Processing == 
    def img_processing(self, frame):
        # Noise Remove
        blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # BGR -> HSV
        hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

        # Color selection
        white_lower_thres = np.uint8([0, 0, 200])
        white_upper_thres = np.uint8([255, 30, 255])
        
        yellow_lower_thres = np.uint8([20, 100, 100])
        yellow_upper_thres = np.uint8([30, 255, 255])
        
        white_mask = cv2.inRange(hsv_frame, white_lower_thres, white_upper_thres)
        yellow_mask = cv2.inRange(hsv_frame, yellow_lower_thres, yellow_upper_thres)
        
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        masked_img = cv2.bitwise_and(frame, frame, mask=mask)
        masked_img = masked_img[:, :, 2]  # 3개의 channel -> 1개의 channel로(gray scale)
                
        return masked_img
           
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
            roi_frame = self.roi(processed_frame)
            bed_frame, inverse_mat = self.bed(roi_frame)
            left_fit_1, right_fit_1 = self.track_lanes_initialize(bed_frame)
            left_fit, right_fit = self.track_lanes_update(bed_frame, left_fit_1, right_fit_1)
            colored_lane = self.lane_fill_poly(bed_frame, frame, left_fit, right_fit, inverse_mat)
            
            cv2.imshow('Result', colored_lane)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = "drive.mp4"
    lane_detection_system = LaneDetection(video_path)
    lane_detection_system.run()
    
if __name__ == "__main__":
    main()
