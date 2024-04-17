import cv2

input_file = r'C:\Users\User\Desktop\MORARI 4월부터 차선검출 코드\LaneDetection\no_gps#5.avi'
output_file = 'output.mp4'

cap = cv2.VideoCapture(input_file)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
