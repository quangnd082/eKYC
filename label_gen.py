import cv2
import os
import glob
import numpy as np


def cut_img(path_to_image):

    """
        input: path to the image
        output: - image after respective transform 
                - coordinate of initial image

        parameter need to change: ###
    """

    img1 = cv2.imread(path_to_image)
    img_copy2 = img1.copy()

    gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 25, 30, 30) ###

    #Chọn ngưỡng thích hợp để tách sách khỏi nền
    _, thresh = cv2.threshold(gray, 191, 255, cv2.THRESH_TOZERO) ###

    #Tìm đường bao của quyển sách
    contours, hyerachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #Lấy 10 contours lớn nhất

    #Tìm ra 4 điểm của contours lớn nhất
    biggest_points = np.array([])
    max_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if hyerachy[0][i][3]==-1 and area > max_area:
            peri = cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], 0.02 * peri, True)
            biggest_points = approx
            max_area = area

    cv2.drawContours(img_copy2, [biggest_points], -1, (0,255,0), 5)
    biggest_points = biggest_points.reshape(len(biggest_points), 2) 

    sum_ = np.sum(biggest_points, axis=1)
    top_left = biggest_points[np.argmin(sum_)]
    bot_right = biggest_points[np.argmax(sum_)]

    diff_ = np.diff(biggest_points, axis=1)
    top_right = biggest_points[np.argmin(diff_)]
    bot_left = biggest_points[np.argmax(diff_)]

    points_needed = [top_left, bot_left, bot_right, top_right]

    # top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    # bottom_width = np.sqrt(((bot_right[0] - bot_left[0]) ** 2) + ((bot_right[1] - bot_left[1]) ** 2))
    # left_height = np.sqrt(((top_left[0] - bot_left[0]) ** 2) + ((top_left[1] - bot_left[1]) ** 2))
    # right_height = np.sqrt(((top_right[0] - bot_right[0]) ** 2) + ((top_right[1] - bot_right[1]) ** 2))

    # max_width = max(int(bottom_width), int(top_width))
    # max_height = max(int(left_height), int(right_height))

    max_width = 450
    max_height = int(max_width*54/86)

    input_points = np.float32([top_left, bot_left, bot_right, top_right])
    output_points = np.float32([[0, 0],
                            [0, max_height - 1],
                            [max_width - 1, max_height - 1],
                            [max_width - 1, 0]])

    matrix = cv2.getPerspectiveTransform(input_points, output_points)
    img_output = cv2.warpPerspective(img1, matrix, (max_width, max_height))

    return [img_output, points_needed]

#create new path containing the image
path = f'cutted_img/image/background_5'
path2 = f'cutted_img/coordinates/background_5'
if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(path2):
    os.makedirs(path2)

img_path = f"result/background_5.jpg/*"

for i in sorted(glob.glob(img_path), key=os.path.getmtime):
    result = cut_img(i)
    # print(i)
    new_img = result[0]
    label_path = i.split("/")[-1]
    path_to_image = path + "/" + label_path
    cv2.imwrite(path_to_image, new_img)
    
    points = result[1]
    path_to_label =  path2 + "/" + label_path + ".txt"
    with open(path_to_label, 'w') as f:
        # flatten points to 1D array
        points = np.array(points).flatten()
        # convert to string
        coodinates = ''
        for i in range(len(points)):
            coodinates += str(points[i]) + " "
        
        coodinates = coodinates[:-1]

        # write to file
        f.write(coodinates)