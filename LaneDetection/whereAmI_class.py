import cv2
import numpy as np
import time
from enum import IntEnum

VIDEO_PATH = r"C:\Users\User\Desktop\성민이 깃헙\simulation-autonomousDriving\LaneDetection\no_gps_obstacle#13.avi"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("=" * 20)
    print("Camera Open Failed")
    print("=" * 20)
    exit()

_, frame = cap.read()


class LaneDetectInfo(IntEnum):
    LANE_1 = 1
    LANE_2 = 2
    LANE_3 = 3
    POSSIBLE_CROSSING_CENTERLINE = 4
    ENCROACHING_CENTERLINE = 5
    CROSSING_PEDESTRIAN_CROSSWALK = 6


class WhereAmI:
    def __init__(self):
        # pixel white, yellow
        self.lower_white = np.array([0, 0, 200], dtype=np.uint8)
        self.upper_white = np.array([180, 30, 255], dtype=np.uint8)

        self.lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        self.upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

        # putText argument
        self.position = (640, 320)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_color = (0, 0, 255)
        self.line_type = 3

        # roi
        self.x1, self.y1, self.width1, self.height1 = int(frame.shape[1] * 0.20), int(frame.shape[0] * 0.8), 300, 200  # 왼쪽
        self.x2, self.y2, self.width2, self.height2 = int(frame.shape[1] * 0.55), int(frame.shape[0] * 0.8), 300, 200  # 오른쪽
        self.thresh_center_x3, self.thresh_center_y3, self.width3, self.height3 = int(frame.shape[1] * 0.15 + 400), int(frame.shape[0] - 150), 50, 150  # 가운데
        self.crosswalk_center_x4, self.crosswalk_center_y4, self.width4, self.height4 = 0, int(frame.shape[0] * 0.7), int(frame.shape[1]), int(frame.shape[0] - frame.shape[0] * 0.7)  # 횡단보도

        # 1차선일 때, 중앙선 침법 threshold
        self.crossing_centerline_thres = 10

        # default, start_time은 횡단보도용
        self.lane_info = LaneDetectInfo.LANE_3
        self.temp_lane_info = None
        self.start_time = time.time()
        self.flag = False

    # 동영상 저장
    def video_record(self):
        output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        record_method = cv2.VideoWriter(f'output_GPS13#_NoObstacle-case2.mp4', fourcc, fps, (output_width, output_height))
        return record_method

    # 횡단보도 검출, output은 윤곽
    def find_contour(self, roi_img, min_area=1500):
        gray = cv2.cvtColor(roi_img.copy(), cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.bilateralFilter(gray, d=3, sigmaColor=10, sigmaSpace=5)
        thresh = cv2.threshold(blur_frame, 200, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crosswalk_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) == 4 and min_area < area:
                crosswalk_contours.append(contour)

        return crosswalk_contours

    # 해당 roi imgs
    def roi_imgs(self, frame):
        roi1 = frame[self.y1:self.y1 + self.height1, self.x1:self.x1 + self.width1]
        roi2 = frame[self.y2:self.y2 + self.height2, self.x2:self.x2 + self.width2]
        roi3 = frame[self.thresh_center_y3:self.thresh_center_y3 + self.height3, self.thresh_center_x3:self.thresh_center_x3 + self.width3]
        roi4 = frame[self.crosswalk_center_y4:self.crosswalk_center_y4 + self.height4, self.crosswalk_center_x4:self.crosswalk_center_x4 + self.width4]

        return roi1, roi2, roi3, roi4

    # == white, yellow, above pixel들 개수 세기 ==
    def count_white_pixel(self, roi_frame):
        white_pixel_num = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV), self.lower_white, self.upper_white))
        return white_pixel_num

    def count_yellow_pixel(self, roi_frame):
        yellow_pixel_num = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV), self.lower_yellow, self.upper_yellow))
        return yellow_pixel_num

    def count_above_yellow_pixel(self, roi_frame):
        above_yellow_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV), self.lower_yellow, self.upper_yellow))
        return above_yellow_pixel

    # 확인차 roi img들에 rectangle 그리는 것
    def draw_rectangle(self, frame):
        cv2.rectangle(frame, (self.x1, self.y1), (self.x1 + self.width1, self.y1 + self.height1), (0, 255, 0), 2)
        cv2.rectangle(frame, (self.x2, self.y2), (self.x2 + self.width2, self.y2 + self.height2), (0, 255, 0), 2)
        cv2.rectangle(frame, (self.thresh_center_x3, self.thresh_center_y3), (self.thresh_center_x3 + self.width3, self.thresh_center_y3 + self.height3), (0, 255, 0), 2)
        cv2.rectangle(frame, (self.crosswalk_center_x4, self.crosswalk_center_y4), (self.crosswalk_center_x4 + self.width4, self.crosswalk_center_y4 + self.height4), (0, 255, 0), 2)

        return frame

    # 몇 차선인지 검출
    def detect(self, frame):
        roi1, roi2, roi3, roi4 = self.roi_imgs(frame)

        left_white_pixel = self.count_white_pixel(roi1)
        left_yellow_pixel = self.count_yellow_pixel(roi1)
        right_white_pixel = self.count_white_pixel(roi2)
        right_yellow_pixel = self.count_yellow_pixel(roi2)

        self.draw_rectangle(frame)

        crosswalk_exist = self.find_contour(roi4)

        # 차선 : 노, 흰, 흰, 노
        if not self.flag:
            if self.lane_info in {LaneDetectInfo.LANE_1, LaneDetectInfo.ENCROACHING_CENTERLINE}:
                above_pixel_cnts = self.count_above_yellow_pixel(roi3)

            # 횡단보도가 아닐 시
            if len(crosswalk_exist) > 0 and len(crosswalk_exist) < 10:
                if self.lane_info == LaneDetectInfo.LANE_3:
                    if right_white_pixel > right_yellow_pixel:
                        self.lane_info = LaneDetectInfo.LANE_2

                elif self.lane_info == LaneDetectInfo.LANE_2:
                    if left_white_pixel < left_yellow_pixel:
                        self.lane_info = LaneDetectInfo.LANE_1
                    elif right_white_pixel < right_yellow_pixel:
                        self.lane_info = LaneDetectInfo.LANE_3

                elif self.lane_info == LaneDetectInfo.LANE_1:
                    if left_white_pixel > left_yellow_pixel:
                        self.lane_info = LaneDetectInfo.LANE_2
                    elif above_pixel_cnts >= self.crossing_centerline_thres:
                        self.lane_info = LaneDetectInfo.POSSIBLE_CROSSING_CENTERLINE

                # heading이 왼쪽으로 꺾이면, 중앙선을 바라보는 것으로 노란색 검출
                elif self.lane_info == LaneDetectInfo.POSSIBLE_CROSSING_CENTERLINE:
                    if right_white_pixel > right_yellow_pixel:
                        self.lane_info = LaneDetectInfo.LANE_1
                    elif right_white_pixel < right_yellow_pixel:
                        self.lane_info = LaneDetectInfo.ENCROACHING_CENTERLINE

                # 역주행 차선들 기점으로, 어차피 중앙선을 넘으면 실격이니 3차선에 있다고만 가정
                elif self.lane_info == LaneDetectInfo.ENCROACHING_CENTERLINE:
                    if above_pixel_cnts >= self.crossing_centerline_thres:
                        self.lane_info = LaneDetectInfo.POSSIBLE_CROSSING_CENTERLINE
                    elif right_white_pixel > right_yellow_pixel:
                        self.lane_info = LaneDetectInfo.LANE_1

                # 이는 하드코딩을 위한 것으로, 횡단보도를 계속 유지하고 있다는 것을 유지(차선이 점선)
                elif self.lane_info == LaneDetectInfo.CROSSING_PEDESTRIAN_CROSSWALK:
                    if 2 <= len(crosswalk_exist) < 4:
                        self.lane_info = self.temp_lane_info

            # 횡단보도 검출
            elif len(crosswalk_exist) >= 10:
                if self.lane_info != LaneDetectInfo.CROSSING_PEDESTRIAN_CROSSWALK:
                    self.temp_lane_info = self.lane_info
                self.lane_info = LaneDetectInfo.CROSSING_PEDESTRIAN_CROSSWALK
                self.start_time = time.time()
                self.flag = True

        # 횡단보도이면 하드코딩으로 1초간 유지
        elif self.flag:
            self.lane_info = LaneDetectInfo.CROSSING_PEDESTRIAN_CROSSWALK
            if time.time() - self.start_time >= 1:
                self.flag = False

        return frame, self.lane_info

    def run(self):
        record = self.video_record()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("=" * 20)
                print("No Frame")
                print("=" * 20)
                break

            update_frame, lane_info = self.detect(frame)
            cv2.putText(update_frame, lane_info.name, self.position, self.font, self.font_scale, self.font_color, self.line_type)
            record.write(update_frame)
            cv2.imshow("Result", update_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                print("=" * 20)
                print("KeyBoard Interrupt")
                print("=" * 20)
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    hee = WhereAmI()
    hee.run()


if __name__ == "__main__":
    main()
