import os
import json


path = r'C:\Users\User\Desktop\gogo\car_labelcomp'
txt_dir = r'C:\Users\User\Desktop\gogo\car_labelcomp.txt'

if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)

file_list = os.listdir(path)
print(f'file_list length : {len(file_list)}')

# json 확장자만
json_file_lists = []
for file_name in file_list:
    if file_name.endswith('.json'):
        json_file_lists.append(file_name)

print(f'json_file_list length: {len(json_file_lists)}')

# 이미지 크기는 640x480(width, height)임
HEIGHT = 480
WIDTH = 640

for json_file in json_file_lists:
    with open(os.path.join(path, json_file), "r") as file:
        json_data = json.load(file)

        txt_filename = os.path.splitext(json_file)[0] + '.txt'
        for shape in json_data["shapes"]:
            class_id = 3 # green : 1, red : 3
            left_top = shape['points'][0] # 좌/상단
            right_bottom = shape['points'][1] # 우/하단

            if left_top > right_bottom:
                temp = left_top
                left_top = right_bottom
                right_bottom = temp
                
            print(f'left top : {left_top}')
            print(f'right bottom : {right_bottom}')

            # left top
            x_left_top = float(left_top[0]) # x
            y_left_top = float(left_top[1]) # y

            # right bottom
            x_right_bottom = float(right_bottom[0]) # x
            y_right_bottom = float(right_bottom[1]) # y

            bbox_width = x_right_bottom - x_left_top
            bbox_height = y_right_bottom - y_left_top
            
            norm_width = bbox_width / WIDTH
            norm_height = bbox_height / HEIGHT

            ctr_x = x_left_top + bbox_width / 2
            norm_ctr_x = ctr_x / WIDTH

            ctr_y = y_left_top + bbox_height / 2
            norm_ctr_y = ctr_y / HEIGHT

            txt_data = "{} {} {} {} {}".format(str(class_id), str(norm_ctr_x), str(norm_ctr_y), str(norm_width), str(norm_height))

            txt_file_path = os.path.join(txt_dir, txt_filename)

            with open(txt_file_path, "a") as txt_file:
                txt_file.write(txt_data)
            print(f'Saved File {txt_data}')
print('All Saved')