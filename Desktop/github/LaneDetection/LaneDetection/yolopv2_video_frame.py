import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
model.to(DEVICE)

VIDEO_PATH = r"C:\Users\H_\Desktop\yolop\output_GPS13#_NoObstacle-case2.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
if cap is None:
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(f'output_GPS13#_NoObstacle-{time.time()}.mp4', fourcc, fps, (frame_width, frame_height))


transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize((640, 640)),  
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    frame_tensor = transform(frame)
    
    frame_tensor = frame_tensor.unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model(frame_tensor)
    
    seg1 = ll_seg_out[0, 0]
    seg1_np = seg1.cpu().numpy()
    seg1_normalized = (seg1_np * 255).astype(np.uint8)

    seg1_color = cv2.cvtColor(seg1_normalized, cv2.COLOR_GRAY2BGR)
    alpha = 0.3
    beta = 1 - alpha
    resized_frame = cv2.resize(frame, (seg1_color.shape[1], seg1_color.shape[0]))

    fusion = cv2.addWeighted(resized_frame, beta, seg1_color, alpha, 0)
    video_writer.write(fusion)
    cv2.imshow('Frame', fusion)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
