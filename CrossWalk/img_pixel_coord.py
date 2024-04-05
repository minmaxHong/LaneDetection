import cv2

# 마우스 클릭 이벤트 콜백 함수
def mouse_callback(event, x, y, flags, param):
    # 마우스 왼쪽 버튼을 클릭할 때
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"마우스 좌클릭: ({x}, {y})")

# 이미지 파일 읽기
image = cv2.imread("captured_frame1280-720.jpg")

# 윈도우 생성 및 이미지 표시
cv2.namedWindow("image")
cv2.imshow("image", image)

# 마우스 이벤트 콜백 함수 등록
cv2.setMouseCallback("image", mouse_callback)

# 키 입력 대기
cv2.waitKey(0)

# 윈도우 종료
cv2.destroyAllWindows()