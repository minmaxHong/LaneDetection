import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"마우스 좌클릭: ({x}, {y})")

image = cv2.imread("captured_frame1280-720.jpg")

cv2.namedWindow("image")
cv2.imshow("image", image)
cv2.setMouseCallback("image", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()