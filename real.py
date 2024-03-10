import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

class LaneDetection:
    def __init__(self, video_path, window_search = 9):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.vertices = None
        self.window_search = window_search
        
        
    # == Img Processing == 
    def img_processing(self, frame):
        # Noise Remove
        blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # RGB -> HSV
        hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_RGB2HSV)

        # Color selection
        white_lower_thres = np.uint8([0, 0, 200])
        white_upper_thres = np.uint8([255, 30, 255])
        
        yellow_lower_thres = np.uint8([20, 100, 100])
        yellow_upper_thres = np.uint8([30, 255, 255])
        
        white_mask = cv2.inRange(hsv_frame, white_lower_thres, white_upper_thres)
        yellow_mask = cv2.inRange(hsv_frame, yellow_lower_thres, yellow_upper_thres)
        
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        masked_img = cv2.bitwise_and(frame, frame, mask = mask)
        masked_img = masked_img[:, :, 2] # 3->1로 dimension고쳐야함 그래야지 track_lane_frame이 작동
        return masked_img
        
    # == Region Of Interest == 
    def roi(self, frame):
        mask = np.zeros_like(frame)
        
        if len(frame.shape) > 2:
            channel_cnt = frame.shape[2]
            ignore_mask_color = (255, ) * channel_cnt
            
        else:
            ignore_mask_color = 255
        
        # bottomleft, topleft, topright, bottomright 순서
        height, width = frame.shape[:2]
        
        bottom_left = [width * 0.1, height * 0.95]
        top_left = [width * 0.4, height * 0.6]
        bottom_right = [width * 0.9, height * 0.95]
        top_right = [width * 0.6, height * 0.6]
        
        self.vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype = np.int32) 
        
        cv2.fillPoly(mask, self.vertices, ignore_mask_color)
        masked_img = cv2.bitwise_and(frame, mask)
        
        return masked_img        
    
    # == bird-eye-view == 
    def bed(self, frame):
        height, width = frame.shape[:2]
        
        bottom_left, top_left, top_right, bottom_right = np.float32(self.vertices[0])

        src_points = np.float32([bottom_left, top_left, top_right, bottom_right])
        dst_points = np.float32([[0, height], [0, 0], [width, 0], [width, height]])
        mat = cv2.getPerspectiveTransform(src_points, dst_points)

        bed_frame = cv2.warpPerspective(frame, mat, (width, height))
        
        return bed_frame
    
    # == Left Lines & Right Lines == 
    def track_lanes_initialize(self, frame):
        histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)
        # plt.clf()
        # plt.plot(histogram)
        # plt.draw()
        # plt.pause(0.01)
        
        out_img = np.dstack((frame, frame, frame)) * 255
        
        mid_point = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:mid_point])
        rightx_base = np.argmax(histogram[mid_point:]) + mid_point  # 절대적값으로 바꿔주기 위함
        
        window_height = int(frame.shape[0] / self.window_search)
        
        nonzero = frame.nonzero() # [행(height)], [열(width)]
        nonzero_y = np.array(nonzero[0]) # height index
        nonzero_x = np.array(nonzero[1]) # width index
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        margin = 100
        minpix = 50
        
        # index값들
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(self.window_search):
            window_y_low = int(frame.shape[0] - (window + 1) * window_height)
            window_y_high = int(frame.shape[0] - window * window_height)
            
            window_x_left_low = leftx_current - margin
            window_x_left_high = leftx_current + margin
            
            window_x_right_low = rightx_current - margin
            window_x_right_high = rightx_current + margin
            
            cv2.rectangle(out_img, (window_x_left_low, window_y_low), (window_x_left_high, window_y_high), (0, 255, 0), 3)
            cv2.rectangle(out_img, (window_x_right_low, window_y_low), (window_x_right_high, window_y_high), (0, 255, 0), 3)
            
            good_left_inds = ((nonzero_y >= window_y_low) &
                               (nonzero_y < window_y_high) &
                                 (nonzero_x >= window_x_left_low) &
                                   (nonzero_x <= window_x_left_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= window_y_low) &
                               (nonzero_y < window_y_high) &
                               (nonzero_x >= window_x_right_low) &
                               (nonzero_x <= window_x_right_high)).nonzero()[0]
            
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # 예외처리 신경써야함
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzero_x[good_left_inds]))
            else:
                leftx_current = leftx_base

            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzero_x[good_right_inds]))
            else:
                rightx_current = rightx_base

        return out_img
    
    
    # == 실행 == 
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
            bed_frame = self.bed(roi_frame)
            track_lane_frame = self.track_lanes_initialize(bed_frame)
            
            # combine_frame = np.concatenate((bed_frame, roi_frame), axis = 1)
            # combine_frame = cv2.resize(combine_frame, (1080, 840), interpolation = cv2.INTER_AREA)
            # cv2.imshow('Combined Frame', combine_frame)
            
            # cv2.imshow('Bed Frame', bed_frame)
            cv2.imshow('Track', track_lane_frame)
            print(f'processed_frame shape : {processed_frame.shape}')
            
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
