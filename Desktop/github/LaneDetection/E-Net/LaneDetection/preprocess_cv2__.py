#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import cv2

from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_real_time_data(img, transform):
    img = Image.fromarray(img)
    img = transform(img)
    return img

def region(image):

    h, w = image.shape #256, 512

    # rectangle = np.array([[(w // 8, h), (w // 8 * 7, h),
    #                        (w // 3 * 2, h // 3), (w // 3, h // 3)]])
    # rectangle = np.array([[(0, h // 4 * 3), (0, h), (w, h), (w, h // 4 * 3),
    #                        (w // 4 * 3, int(h * 0.63)), (w // 4, int(h * 0.63))]])
    rectangle = np.array([[(0, h // 4 * 3), (0, h), (w, h), (w, h // 4 * 3),
                             (w // 4 * 3, int(h*0.5)), (w // 4, int(h *0.5))]]) #ROI 영역 변형하는 부분!
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, rectangle, 255) #ROI영역 설정
    mask = cv2.bitwise_and(image, mask) #ROI와 bitwise연산

    return mask


def getImages(img, model):
    # img는 resize된 
    # if os.path.isdir(os.getcwd() + "/SNU_DATASET/test") == False:
    #     os.mkdir(os.getcwd() + "/SNU_DATASET/test")

    ########################################################
    # args = parse_args()
    # img 원본은 320x180
    # img_path = "./SNU_DATASET"
    resize_height = 256 # args.height
    resize_width = 512 # args.width #resize: 512x256 -> image 크기 확장

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) #resize -> tensor -> Normalize

    model.eval() #평가모드
    model.to(DEVICE) #gpu에 올리기

    # img를 frame -> transfrom 함수로 고쳐야함
    dummy_input = load_real_time_data(img, data_transform).to(DEVICE) #img를 PIL로 바꾼 후에 transform을 진행함 -> 이후 나온 결과를 cuda에
    dummy_input = torch.unsqueeze(dummy_input, dim=0) #차원을 하나 증가시켜줌
    outputs = model(dummy_input) #model에 넣어서 결과 추출 -> dictionary 형태로 결과 저장
    """
    {
        'instance_seg_logits': pix_embedding,
        'binary_seg_pred': binary_seg_ret,
        'binary_seg_logits': binary
    }
    """

    # input = Image.fromarray(img)
    # input = input.resize((resize_width, resize_height))
    # input = np.array(input)
    # 이 과정은 도대체 왜 하는지 모르겠음 -> 주석처리
    # chatgpt피셜: 이 과정을 거치고 resize하나, 바로 resize를 하나 차이는 없다 -> img로 변경!
    
    # numpy값
    instance_pred = torch.squeeze(
         outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    
    binary_pred = torch.squeeze(
        outputs['binary_seg_pred']).to('cpu').numpy() * 255
    
    cv2.imwrite("./test_img/binary_pred.jpg", binary_pred) 
    
    #################### Preprocessing : crop ########################
    binary = np.array(binary_pred, dtype=np.uint8) #cv2.imshow를 위해서는 uint8이 필요
    cv2.imshow('original_segmentation',binary)
    mask = region(binary) #ROI를 실행한 차선
    cv2.imwrite("./test_img/mask.jpg", mask)
    
    input = cv2.resize(img, (320, 180))
    mask = cv2.resize(mask, (320, 180))

    return input, mask
    
    # black = [0, 0, 0]

    # constant = cv2.copyMakeBorder(mask, 135, 135, 240, 240, cv2.BORDER_CONSTANT, value=black) #경계를 추가하고
    # constant = cv2.resize(constant, (320, 180)) #resize

    # bordered_color = cv2.copyMakeBorder(input, 135, 135, 240, 240, cv2.BORDER_CONSTANT, value=black) #input도 경계를 추가하고
    # bordered_color = cv2.resize(bordered_color, (320, 180)) #resize
    # bordered_color = cv2.cvtColor(bordered_color, cv2.COLOR_BGR2RGB) -> 이미 RGB로 되어있으므로 필요없음
    
    # cv2.imwrite("./bordered_color.jpg", bordered_color)
    # cv2.imwrite("./constant.jpg", constant)
    
    # return input, bordered_color, mask
    # 모두 320x180 image인데 input은 원래 이미지, bordered_color는 input image에 검은색 액자를 씌워서 resize한 image(어차피 안씀)
    # constant는 ROI를 이용해서 차선을 추출한 binary image를 검은색 액자를 씌워서 resize한 image


def perspective_warp(img,
                     dst_size=(320, 180),
                     src=np.float32([(0.28, 0.55), (0.72, 0.55), (0.04, 0.95), (0.96, 0.95)]), # 전체를 변환하는 것이 아닌 중간을 변환함
                     dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
                    # (0.35, 0.63), (0.65, 0.63), (0.08, 0.75), (0.92, 0.75) alomst best but not in highly curved lane
                    # (0.395, 0.58), (0.605, 0.58), (0.06, 0.75), (0.94, 0.75)
    cv2.imshow('i want show', img)
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    # For destination points, I'm arbitrarily choosing some points  to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def inv_perspective_warp(img,
                         dst_size=(320, 180),
                         src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                         dst=np.float32([(0.28, 0.55), (0.72, 0.55), (0.04, 0.95), (0.96, 0.95)])):
                                        # (0.415, 0.58), (0.585, 0.58), (0.06, 0.75), (0.94, 0.75)
                                        # (0.36, 0.54), (0.64, 0.54), (0, 0.85), (1, 0.85)
                                        # 0.35, 0.55
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:, :], axis=0) # 중간부터 밑까지 열 방향으로 쭉 더하기 shape:(320,)
    # hist[img.shape[1]//5*4:] = 0 #계산한 히스토그램에 대하여 이미지 오른쪽의 20%를 모두 0으로 만듬
    return hist

def sliding_window(img, nwindows=9, margin=30, minpix=1, draw_windows=True):
    # global left_a, left_b, left_c,right_a, right_b, right_c
    # img는 (180, 320)
    left_a, left_b, left_c = [], [], []
    right_a, right_b, right_c = [], [], []

    left_fit_ = np.empty(3) #예상할 수 없는 3개의 값이 들어감 -> 3개의 벡터
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255 # 똑같은 3개의 값을 0아니면 255의 값으로 3채널 변환(각각의 채널들은 같은 값을 가짐)
    histogram = get_hist(img) # (320,)의 열벡터
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2) # 절반으로 나눈값 (160)
    leftx_base = np.argmax(histogram[:midpoint]) # 0~159중 가장 큰 값
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # 160~319중 가장 큰 값 index 반환

    # Set height of windows
    window_height = int(img.shape[0]/nwindows) #window의 높이 지정
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero() # 0이 아닌 index값 반환 -> 0, 1번 index 따로 따로 저장됨 -> width x height의 개수가 나올 수 있음
    nonzeroy = np.array(nonzero[0]) # 위에서 저장된 0이 아닌 y축 index
    nonzerox = np.array(nonzero[1]) # 위에서 저장된 0이 아닌 x축 index
    # Current positions to be updated for each window
    leftx_current = leftx_base # 현재 값에 가장 큰 index 저장
    rightx_current = rightx_base # 현재 값에 가장 큰 index 저장
    

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = [] # 왼쪽, 오른쪽에 대한 각각의 차선 index 저장할 list

    # Step through the windows one by one
    for window in range(nwindows):
        # nwindows만큼 반복을 진행하는데 여기서 window는 index 역할을 함 (몇 번째 window값인지)
        # 위에서부터 0번, 1번, 2번 ... window
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height # window의 밑변 위치
        win_y_high = img.shape[0] - window*window_height # window의 윗변 위치
        win_xleft_low = leftx_current - margin # 열의 가장 큰 값에서 margin만큼 빼기
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # 위의 6개의 변수들은 각각의 사각형 꼭짓점의 좌표들임
        # Draw the windows on the visualization image, 꼭짓점의 좌표들을 이용해서 사각형 그리기
        if draw_windows == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), # 왼쪽 박스
                          (100, 255, 255), 3) # (이미지 파일, 시작 좌표, 끝나는 좌표, 색상, 두께) -> out_img에 박스 집어 넣기
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), # 오른쪽 박스
                          (100, 255, 255), 3) 
        # Identify the nonzero pixels in x and y within the window
        # window 값에 맞는 index값을 저장해줌
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds) # 좋은 차선들을 모아둔 list
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix: # minpix(하나)이상 검출했다면
            leftx_current = int(np.mean(nonzerox[good_left_inds])) # good_left_inds에 해당하는 index에 nonzerox값의 평균을 낸 정수값을 leftx_current에 저장
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            
        #leftx_current와 rightx_current는 정말 잘 검출된 좋은 값을 가짐 (왼쪽과 오른쪽 차선의 차선의 좌우<->에 대한 최적의 값)
            
        # print("good_left_inds_len:", len(good_left_inds))
        # print("good_right_inds_len :", len(good_right_inds))


#        if len(good_right_inds) > minpix:
#            rightx_current = np.int(np.mean([leftx_current +900, np.mean(nonzerox[good_right_inds])]))
#        elif len(good_left_inds) > minpix:
#            rightx_current = np.int(np.mean([np.mean(nonzerox[good_left_inds]) +900, rightx_current]))
#        if len(good_left_inds) > minpix:
#            leftx_current = np.int(np.mean([rightx_current -900, np.mean(nonzerox[good_left_inds])]))
#        elif len(good_right_inds) > minpix:
#            leftx_current = np.int(np.mean([np.mean(nonzerox[good_right_inds]) -900, leftx_current]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds) # 기존에 구한 차선 값들을 합치기

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] # 계산된 각각의 차선의 값에 대한 원래 값들의 index들 (원래 차선 값) -> 짝지어 있음

    leftDetected = False
    rightDetected = False
    direction = 0 # 1 : right, -1 : left
    balanced = 0 # 1 : right, -1 : left    
    
    """
    ::TODO::
    
    left_fit[0]과 right_fit[0] 이 같은 부호라면 detected = 1 아니라면 0
    
    
    """
    
    # Fit a second order polynomial to each
    # left fit
    try:
        left_fit = np.polyfit(lefty, leftx, 2) #(lefty, leftx)에 대한 2차 방정식의 계수를 반환함
        
        left_a.append(left_fit[0]) # 2차식 계수
        left_b.append(left_fit[1]) # 1차식 계수 
        left_c.append(left_fit[2]) # 상수항 -> 1개 일 수 밖에 없음..!
        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:]) # 따라서 마지막 10개의 평균을 구하는 것은 의미가 없음 -> 그냥 수 저장으로 바꿔도 될 듯
        # left_a = left_fit[0] = left_fit_[0] 세 값이 모두 같음

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0]) # 0부터 img.shape[0]-1까지 img.shape[0]개의 점을 생성 -> 그냥 0부터 정수로 179까지 1씩 증가함
        
        left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2] # 왼쪽 차선에 대해서 2차 함수 생성

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100] # 차선에 해당하는 y값과 x값에 [255, 0, 100]의 color 값을 대입함 -> 차선 색칠

        leftDetected = True # 왼쪽 차선 검출 완료
        
    except:
        pass
    
    # right fit
    try:    
        # 오른쪽 차선도 똑같이 진행
        right_fit = np.polyfit(righty, rightx, 2)
        

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])
        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        
        right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

        out_img[nonzeroy[right_lane_inds],nonzerox[right_lane_inds]] = [0, 100, 255] # 차선에 해당하는 y값과 x값에 [0, 100, 255]의 color 값을 대입함

        rightDetected = True
        
    except:
        pass
        
    

    # 생성된 곡선은 왔다리 갔다리~ 궁금하면 밑에 주석 출력해 볼 것 (q 한 번씩 누르기)
    # plt.figure()
    # plt.plot(ploty, left_fitx)
    # plt.plot(ploty, right_fitx)
    # plt.show()
    
    
    # if len(lefty) > 20 and len(righty) > 20:
    #     if 150 < max(lefty) < 160 and 150 < max(righty) < 160:
    #         if leftx[0] > (320 - rightx[0]):
    #             balanced = 1 # have to MOVE left
    #         elif leftx[0] < (320 - rightx[0]):
    #             balanced = -1
    #     elif 150 < max(lefty) < 160:
    #         balanced = -1
    #     elif 150 < max(righty) < 160:
    #         balanced = -1
        
    # Which direction the curved lane made
    if leftDetected and rightDetected: # 차선을 둘 다 잘 검출한 경우
        # leftx[0]과 rightx[0]은 아마 제일 처음 검출되는 차선이지 않을까 싶음 -> 가로 세로 방향 <->
        if 150 < max(lefty) < 160 and 150 < max(righty) < 160: # 왼쪽, 오른쪽 각각의 검출된 차선의 y의 index의 최대값이 150과 160 사이라면 -> 바로 앞에 차선이 있음을 의미
            if abs(leftx[0] - (320 - rightx[0])) < 10: # 320 - rightx[0]은 반전한 값인 것 같음 -> leftx[0]과의 비교를 위해서, 얼마나 대칭이냐
                pass
            elif leftx[0] > (320 - rightx[0]): # left가 더 왼쪽에 있다면
                balanced = 320 - rightx[0] # have to MOVE left -> 320 - rightx[0]만큼 더 가야함
            elif leftx[0] < (320 - rightx[0]): # right가 더 오른쪽에 있다면
                balanced = -leftx[0] # have to MOVE right -> -leftx[0]만큼 더 가야함
            # 의문점: 왼쪽으로 가야하는 것은 320 - rightx[0]인데 오른쪽으로 가야하는 것은 왜 -leftx[0]인 음수인가?
            # balanced는 차량 balance를 말하는 것인가?

        elif 150 < max(lefty) < 160:
            balanced = -leftx[0]
        elif 150 < max(righty) < 160:
            balanced = 320 - rightx[0]
        else:
            if abs(max(lefty) - max(righty)) < 10:
                pass
            elif max(lefty) - max(righty) > 0:
                balanced = -leftx[0]
            elif max(lefty) - max(righty) < 0:
                balanced = 320 - rightx[0]
                
        if max(leftx) - min(leftx) != 0 and max(rightx) - min(rightx) != 0:
            a = (max(lefty) - min(lefty)) / (min(leftx) - max(leftx)) # + : right, - : wrong
            b = (min(righty) - max(righty)) / (min(rightx) - max(rightx)) # + : right, - : wrong
            
            if a < b:
                k = -a
            else:
                k = b
                
            if a != 0 and b != 0:
                _balanced = 1 / b + 1 / a
                # if a > b:
                #     balanced = 1 / b - 1 / a
                # else:
                #     balanced = 1 / (b - a)
            else:
                _balanced = 0 
        else:
            k = 0
            _balanced = 0 

        balanced += 50 * _balanced
        
        return out_img, (left_fitx, right_fitx), balanced, True # "a : {:2f}\n b : {:2f}".format(float(a), float(b))
    
    elif leftDetected: # or rightDetected
        
        # if len(leftx) > 10:
            
        # elif len(rightx) > 10:
            
        if leftx[0] > leftx[-1]:
            direction = -1
        else:
            direction = 1
        
        # if left_fit[0] > 0:
        #     direction = -1
        # else:
        #     direction = 1
            
        if leftx[0] > 30:
            balanced = -leftx[0]
            
        if max(leftx) - min(leftx) != 0:
            k = (min(lefty) - max(lefty)) / (max(leftx) - min(leftx)) # + : right, - : wrong
        else:
            k = 0
        
        # if k < 0:
        #     balanced = -balanced
        
        if k != 0:
            _balanced = 1 / k
        else:
            _balanced = 0
        
        if k > 0:   
            # _balanced = -k
            balanced -= 100 * _balanced
        else:
            # _balanced = k
            # balanced = 320 - leftx[0] - 100 * _balanced
            # balanced = 320 - max(leftx) - 100 * _balanced
            balanced = -85
            
        return out_img, (left_fitx, 0), balanced, False
    
    elif rightDetected:
        if rightx[0] < rightx[-1]:
            direction = -1
        else:
            direction = 1
        
        # if right_fit[0] > 0:
        #     direction = -1
        # else:
        #     direction = 1
        
        if rightx[0] < 290:
            balanced = 320 - rightx[0]
        
        if max(rightx) - min(rightx) != 0:
            k = (min(righty) - max(righty)) / (min(rightx) - max(rightx)) # + : right, - : wrong
        else:
            k = 0
        
        # if k < 0:
        #     balanced = -balanced

        # balanced = k
        if k != 0:
            _balanced = 1 / k
        else:
            _balanced = 0
        
        if k > 0:   
            # _balanced = k
            balanced += 100 * _balanced
        else:
            # _balanced = k
            # balanced = 320 - rightx[0] + 100 * _balanced
            # balanced = 320 - min(rightx) + 100 * _balanced
            balanced = 85
            
        return out_img, (0, right_fitx), balanced, False
    
    else:
        return out_img, (0, 0), balanced,  False


def draw_lanes(img, left_fit, right_fit): # img랑 각각 차선의 2차 방정식 
    try:
        #print("shape", img.shape) -> (180, 320, 3)

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0]) # 0부터 179까지 1씩 증가
        color_img = np.zeros_like(img) # img 사이즈에 맞게 0으로 채워짐 (180, 320, 3)
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        #flipud : 상하반전, vstack : 상하로 합치기, hstack : 좌우로 합치기 [[]]기준임

        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))]) # 오른쪽 차선도 똑같이 진행
        points = np.hstack((left, right)) # 차선끼리의 배열이 합쳐짐

        cv2.fillPoly(color_img, np.int_(points), (0, 200, 255)) # 그림 그리기
        cv2.imwrite("./test_img/lines_poly.jpg",color_img)

        inv_perspective = inv_perspective_warp(color_img) # birdeyeview 한 것을 원래대로 만들기
        lane_figure = inv_perspective # 원래대로 만든 차선
        cv2.imwrite("./test_img/psp_lines_poly.jpg",inv_perspective)

        inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0) # img와 inv_perspective 사진을 하나로 합치기
        return inv_perspective, lane_figure
        
    except: # 그리지 못하는 경우 -> left나 right 둘 중 하나가 0
        print("Draw Failed!")
        cv2.putText(color_img, "draw_failed", org=(90, 160),fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.6, color=(255 ,255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False) 
        cv2.imwrite("./test_img/lines_poly.jpg",color_img)
        inv_perspective = inv_perspective_warp(img)
        inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
        return inv_perspective, 0
    


def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0]) # 점을 찍음 (0~179)
    y_eval = np.max(ploty) # 179
    # 차선 폭에 대한 상수값. 일반적으로 차선의 폭은 약 3.7m이며 이를 720pixel의 이미지로 표현할 때, 30.5pixel정도가 차선의 폭이 된다.
    ym_per_pix = 14/360  # 30.5/720 # meters per pixel in y dimension 
    # 카메라 시점에 대한 상수값. 일반적으로 차량의 카메라는 차량 중심에 위치하며 바닥면과 카메라 사이의 거리가 고정되어 있다.
    # 바닥면의 기울기나 차량의 이동 등에 따라서 시점이 변할 수 있다.    
    xm_per_pix = 1.75/360  # 3.7/720 # meters per pixel in x dimension

    try:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        # 곡률을 구하는 함수!
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        car_pos = img.shape[1]/2
        l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
        
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        
        center = (car_pos - lane_center_position) * xm_per_pix / 10 - 0.07
        # 가운데가 0.07의 값으로 나옴 -> 왜인지는 잘 모르겠음
        # Now our radius of curvature is in meters
        if center > 0: 
            # print("LEFTLEFTLEFTLEFTLEFTLEFTLEFTLEFT")
            direction = "LEFT" # 왼쪽으로 가야함을 나타내는 의미로 사용
        elif center < 0: 
            # print("RIGHTRIGHTRIGHTRIGHTRIGHTRIGHTRIGHT")
            direction = "RIGHT"
        else:
            # print("CENTERCENTERCENTERCENTERCENTERCENTER")
            direction = "CENTER"

        return (left_curverad, right_curverad, center, direction) # (왼쪽 차선 곡률, 오른쪽 차선 곡률, center값, direction값)
    except:
        return ([0], [0], 0, "False")


def keepCenter(center, now, file=None):
    if len(center) < 1:
        return center[-1], 1    
    else:
        # center.append(now)
        # arr = np.array([center])
        # arr = np.asarray(center, dtype=np.float16)    
        
        
        # print("\n\n\narr[-1] : {}\n\n\n".format(normalized[0]))
        # print("mean :", np.mean(center))
        # print("std :", np.std(center))
        
        if file is not None:
            file.write("mean :{}\n".format(np.mean(center)))
            file.write("std :{}\n\n".format(np.std(center)))
        
        # normalized = normalize(arr)
        # if abs(normalized[0][-1]) > 0.0025:
        if abs(center[-1] - now) > 0.3: # important
                # print("Im AAAAAAAAAAAAAAAA", center[-1])
                return center[-1], -1
            
        if abs(now) > 1:
            if now > 0:
                # print("Im BBBBBBBBBBBBBBBB", 1)
                # return center[-2], -1 # return normalized[-2], -1
                return 1, 1
            else:
                # print("Im CCCCCCCCCCCCCCCC", -1)
                return -1, 1
        else:
            return now, 1  # return normalized[-1], 1