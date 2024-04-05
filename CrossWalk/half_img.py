import cv2

video_path = 'output_clip.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        center = frame.shape[0] // 2

        right_half = frame[:center, :]

        gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.bilateralFilter(gray, d=3, sigmaColor=10, sigmaSpace=5)
        ret, mask = cv2.threshold(blur_frame, 80, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = 100
        crosswalk_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) == 4 and min_area < area:  # 횡단보도는 일반적으로 4개의 꼭지점을 가짐
                crosswalk_contours.append(contour)

        cv2.drawContours(frame, crosswalk_contours, -1, (0, 255, 0), 2)

        cv2.imshow('Combined Frame', mask)
        cv2.imshow('Origin', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# 종료
cap.release()
cv2.destroyAllWindows()
