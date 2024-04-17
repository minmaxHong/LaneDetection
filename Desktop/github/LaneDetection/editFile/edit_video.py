from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2

# == 동영상 총 길이(초) == 
def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_cnt / fps

    return duration

# == 동영상 편집 == 
def make_clip_video(path, save_path, start_t, end_t):
    clip_video = VideoFileClip(path).subclip(start_t, end_t)
    clip_video.write_videofile(save_path, codec='libx264')  # codec을 'libx264'로 지정

video_path = r"C:\Users\User\Desktop\MORARI 4월부터 차선검출 코드\LaneDetection\no_gps#5.mp4"
save_path = "output_clip.mp4"  
start_time = 20  
end_time = get_video_length(video_path) 

make_clip_video(video_path, save_path, start_time, end_time)
