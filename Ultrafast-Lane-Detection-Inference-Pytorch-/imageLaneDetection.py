import cv2
import torch
import psutil
import subprocess
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# 코드 정보
def hardware_check():
    # CPU
    cpu_percent = psutil.cpu_percent()
    cpu_count = psutil.cpu_count()

    # Memory
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 3)  
    available_memory = memory_info.available / (1024 ** 3) 
    used_memory = total_memory - available_memory

    # GPU
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE)
    gpu_usage = result.stdout.decode('utf-8').strip()
    print('='*30)
    print(f'사용 중인 CPU: {cpu_percent}%')
    print(f"전체 메모리: {total_memory:.2f}GB")
    print(f"사용 가능한 메모리: {available_memory:.2f}GB")
    print(f"사용 중인 메모리: {used_memory:.2f}GB")
    print(f'사용 중인 GPU : {gpu_usage}%')
    print('='*30)

def initialize_lane_detector(model_path, model_type, use_gpu=True):
    return UltrafastLaneDetector(model_path, model_type, use_gpu)

def process_video(video_path, lane_detector, output_video_path=None):
    cap = cv2.VideoCapture(video_path)
    
    if output_video_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Use XVID codec for AVI format
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = lane_detector.detect_lanes(frame)
        
        if output_video_path:
            out.write(output_frame)

        hardware_check()
        # output_frame = cv2.resize(output_frame, dsize = (1280, 720))
        cv2.imshow("Detected lanes", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_video_path:
        out.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = r"/home/macaron/바탕화면/홍성민의 작업공간/Hong_github/PT_weight_file/Ultra_culane_18.pth"
    video_path = r"/home/macaron/바탕화면/홍성민의 작업공간/Hong_github/test_data/CLRNet_video.mp4"
    # output_video_path = None
    output_video_path = r"/home/macaron/바탕화면/홍성민의 작업공간/results_file/tunnel_video_ultrafast_CULANE.mp4" 

    model_type = ModelType.CULANE
    use_gpu = torch.cuda.is_available()
    

    lane_detector = initialize_lane_detector(model_path, model_type, use_gpu)
    process_video(video_path, lane_detector, output_video_path)
