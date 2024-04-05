import cv2

input_file = 'input.avi'
output_file = 'output.mp4'

# 비디오 캡처 생성
cap = cv2.VideoCapture(input_file)

# 비디오 프레임 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 인코더 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# 프레임 읽어서 쓰기
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
