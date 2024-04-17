import cv2
import numpy as np

video_path = 'output_clip.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# ======================== 앞에 있는 횡단보도 검출 ==================================
# == img 전처리 == 
def process_img(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.bilateralFilter(gray, d=3, sigmaColor=10, sigmaSpace=5)
    thresh = cv2.threshold(blur_frame, 200, 255, cv2.THRESH_BINARY)[1]

    return thresh

# == ROI 설정 == 
def roi(thresh):
    close_mask = np.zeros_like(thresh)
    
    ignore_mask = 255
    
    height, width = frame.shape[:2]
    
    # 가까이 있는 것
    close_bottom_left = [0, height]
    close_top_left = [0, height * 0.7]
    close_bottom_right = [width, height]
    close_top_right = [width, height * 0.7]
    
    close_vertice = np.array([[close_bottom_left, close_top_left,
                               close_top_right, close_bottom_right]], dtype = np.int32)
    
    # 가까이 있는 것 fillPoly
    cv2.fillPoly(close_mask, close_vertice, ignore_mask)
    close_masked_img = cv2.bitwise_and(thresh, close_mask)
    
    return close_masked_img, close_vertice


# == 사각형 찾기 == 
def find_contour(roi_img, min_area = 1300):
    contours, _ = cv2.findContours(roi_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crosswalk_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) == 4 and min_area < area:  # 횡단보도는 일반적으로 4개의 꼭지점을 가짐
            crosswalk_contours.append(contour)
            
    return crosswalk_contours       

# 횡단보고 검출
def detect_crosswalk(frame):
    thresh = process_img(frame)
    region_interst, close_vertice = roi(thresh)
    crosswalk_contours = find_contour(region_interst)
    
    cv2.drawContours(frame, crosswalk_contours, -1, (0, 255, 0), 2)
    
    # 가까이 있는 것 ROI
    cv2.putText(frame, "Region of Close Crosswalk", close_vertice[0][1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.line(frame, close_vertice[0][0], close_vertice[0][1], (255, 0, 0), 2)
    cv2.line(frame, close_vertice[0][1], close_vertice[0][2], (255, 0, 0), 2)
    cv2.line(frame, close_vertice[0][2], close_vertice[0][3], (255, 0, 0), 2)
    cv2.line(frame, close_vertice[0][3], close_vertice[0][0], (255, 0, 0), 2)
    
    
    if len(crosswalk_contours) > 5:
        print(f'탐지한 Rectangle 개수 : {len(crosswalk_contours)}')
    
    return crosswalk_contours
# ========================================================================================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = detect_crosswalk(frame)
    cv2.drawContours(frame, processed_frame, -1, (0, 255, 0), 2)
    
    cv2.imshow('Persp', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
