import cv2
import numpy as np

video_path = r"C:\Users\User\Desktop\MORARI 4월부터 차선검출 코드\LaneDetection\no_gps#5.mp4"
cap = cv2.VideoCapture(video_path)
if cap is None:
    exit()

# == Img processing == 
def process(frame):
    global white_mask
    global yellow_mask
    
    blur_frame = cv2.bilateralFilter(frame, d=3, sigmaColor=10, sigmaSpace=5)
    hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

    # White
    white_lower_thres = np.uint8([0, 0, 200])
    white_upper_thres = np.uint8([255, 255, 255])
    
    # Yellow
    yellow_lower_thres = np.uint8([20, 100, 100])
    yellow_upper_thres = np.uint8([30, 255, 255])
    
    white_mask = cv2.inRange(hsv_frame, white_lower_thres, white_upper_thres)
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower_thres, yellow_upper_thres)
    
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame

# == Interpolation == 
def interpolation_lane(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.dilate(frame, kernel)
    
    return dst

# == ROI 설정 == 
def roi(frame):
    global right_masked_img
    global left_masked_img
    
    left_mask = np.zeros_like(frame)
    right_mask = np.zeros_like(frame)
    
    if len(frame.shape) > 2:
        channel_cnt = frame.shape[2]
        ignore_mask_color = (255, ) * channel_cnt
        
    else:
        ignore_mask_color = 255
    
    height, width = frame.shape[:2]
    
    
    # 왼쪽 차선
    left_lane_bottom_left = [0, height * 0.95]
    left_lane_top_left = [width * 0.4, height * 0.6]
    left_lane_bottom_right = [width * 0.5, height * 0.95]
    left_lane_top_right = [width * 0.5, height * 0.6]
    
    left_lane_vertices = np.array([[left_lane_bottom_left, left_lane_top_left, left_lane_top_right, left_lane_bottom_right]],
                                dtype=np.int32) 
    
    cv2.fillPoly(left_mask, left_lane_vertices, ignore_mask_color)
    left_masked_img = cv2.bitwise_and(frame, left_mask)
    
    # 오른쪽 차선
    right_lane_bottom_left = [width * 0.6, height * 0.95]
    right_lane_top_left = [width * 0.6, height * 0.6]
    right_lane_bottom_right = [width, height * 0.95]
    right_lane_top_right = [width * 0.7, height * 0.6]
    
    right_lane_vertices = np.array([[right_lane_bottom_left, right_lane_top_left, right_lane_top_right, right_lane_bottom_right]],
                                   dtype = np.int32)
    
    cv2.fillPoly(right_mask, right_lane_vertices, ignore_mask_color)
    right_masked_img = cv2.bitwise_and(frame, right_mask)
    
    result_masked_img = cv2.bitwise_or(left_masked_img, right_masked_img)
    
    return result_masked_img, left_lane_vertices, right_lane_vertices


# == 허프변환을 이용한 차선 검출 == 
def hough_transform_lane_detect(frame, left_or_right=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minLineLength = 20
    maxLineGap = 300
    
    lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    # 왼쪽 차선 검출
    if left_or_right:
        left_lines = []
        left_length = []
        
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:  # (x1, y1) : 시작점, (x2, y2) : 끝점 --> x : width, y : height
                    start_pts = [x1, y1]
                    end_pts = [x2, y2]
                    
                    # width가 같은 것은 필요 없음 (차선은 사선)
                    if start_pts[0] == end_pts[0]:
                        continue
                    
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)  # slope ('inf')가 되는 것을 방지
                    intercept = y1 - (slope * x1)
                    length = np.sqrt(((y2 - y1) ** 2 + (x2 - x1) ** 2))
                    
                    if slope < 0 and slope:
                        left_lines.append((slope, intercept))
                        left_length.append(length)
        
        # 가장 긴 선을 선택
        if len(left_length) > 0:
            max_index = np.argmax(left_length)
            left_lane = left_lines[max_index]
            return left_lane
        else:
            return None
    # 오른쪽 차선 검출은 여기에 구현해야 함
    else:
        pass

# == 허프변환 선 그리기 == 
def draw_hough_lines(frame, lines, color=(0, 0, 255), thickness=3):
    frame = np.copy(frame)
    
    if lines is not None:
        cv2.line(frame, lines[0], lines[1], (0, 0, 255), 5)
                    
    return frame

# ===========================================================================================
# == 왼쪽 차선 검출 == 
def left_lane_detect(frame):
    left_lane = hough_transform_lane_detect(frame)
    
    if left_lane is not None:
        slope, intercept = left_lane
        y1 = int(frame.shape[0])
        y2 = int(frame.shape[0] * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        left_lane_frame = draw_hough_lines(frame, ((x1, y1), (x2, y2)), (0, 0, 255), thickness=3)
        return left_lane_frame
    else:
        return frame  # 왼쪽 차선이 검출되지 않으면 원본 프레임 반환

# ===========================================================================================
while True:
    ret, frame = cap.read()
    
    if ret is None:
        print('No Frame')
        break
    
    # cv2.imshow('ORIGIN', frame)
    
    # 순서 
    # 1. Img Processing
    processed_frame = process(frame)
    # cv2.imshow('1', processed_frame)
    
    # 2. 비어진 부분 interpolation
    interpolation_frame = interpolation_lane(processed_frame)
    # cv2.imshow('2', interpolation_frame)
        
    # 3. ROI 설정
        # - 현재 정해진 차선만
    roi_frame, left_vertices, right_vertices = roi(interpolation_frame)
    cv2.imshow('ROI', roi_frame)
 
    # cv2.imshow('4', roi_frame)
        # - 차선 종류
            # 1. 1차선(왼쪽은 노란색, 오른쪽은 흰색)
            # 2. 2차선(왼쪽/오른쪽 둘다 흰색)
            # 3. 3차선(왼쪽은 흰색, 오른쪽은 노란색)

    # 4. 왼쪽 차선 검출 
    left = left_lane_detect(left_masked_img)
    frame_with_left_lane = cv2.addWeighted(frame, 1, left, 0.5, 0)
    cv2.imshow('LEFT', left), cv2.imshow('frame_with_left', frame_with_left_lane)
    
    # 5. 오른쪽 차선 검출

    
    # 6. 2개의 차선 검출 있을때
        # 왼쪽 차선, 오른쪽 차선 검출해야함
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
