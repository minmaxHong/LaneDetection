import cv2
import numpy as np
import e_net_architect
import torch

model_path = 'ENET_second.pth'
enet_model = e_net_architect.ENet(2, 4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

enet_model.load_state_dict(torch.load(model_path))
enet_model.eval()

def process_and_save_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), 10, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input_image = cv2.resize(input_image, (512, 256))
        input_image = input_image[..., None]
        input_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1)

        with torch.no_grad():
            binary_logits, instance_logits = enet_model(input_tensor.unsqueeze(0))

        binary_seg = torch.argmax(binary_logits, dim=1).squeeze().numpy()
        instance_seg = torch.argmax(instance_logits, dim=1).squeeze().numpy()

        binary_seg_grayscale = np.zeros_like(input_image[:, :, 0])
        binary_seg_grayscale[binary_seg == 1] = 255

        binary_seg_grayscale = cv2.resize(binary_seg_grayscale, (frame_width, frame_height))

        output_image = frame.copy()
        output_image[:, :, 0] = cv2.addWeighted(output_image[:, :, 0], 0.5, binary_seg_grayscale, 0.5, gamma=0)

        out.write(output_image)

        cv2.imshow('Lane Detection', output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_video_path = "video.mp4"
output_video_path = 'output_inference_video.mp4'
process_and_save_video(input_video_path, output_video_path)
