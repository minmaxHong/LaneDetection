import numpy as np
import cv2

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

result = model.predict('test1.jpg', save = True, conf = 0.5)
plots = result[0].plot()

cv2.imshow('Img', plots)
cv2.waitKey()
cv2.destroyAllWindows()