import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# Load image

dir = 'result'
img_list = []

for filename in os.listdir(dir):
    img_list.append(filename)



for i in range(len(img_list)):
    img = cv2.imread(dir + '/' + img_list[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    coodinate = plt.ginput(4)

    # convert to int
    coodinate = np.array(coodinate)
    coodinate = coodinate.astype(int)
    
    # save coodinate to file
    f = open('label/' + str(img_list[i]) + '.txt', 'w')
    for coo in coodinate:
        f.write(str(coo[0]) + ' ' + str(coo[1]) + '\n')
    f.close()

    
