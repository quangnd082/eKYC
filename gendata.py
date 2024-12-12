import cv2
import numpy as np
import os

# Load the original image

original_dir = 'anh1'

original_list = []

for filename in os.listdir(original_dir):
        original_list.append(filename)

# load background image
background_dir = 'background'

background_list = []

for filename in os.listdir(background_dir):
        background_list.append(filename)

# load background_rotate image
background_rotate_dir = 'background1'

background_rotate_list = []

for filename in os.listdir(background_rotate_dir):
        background_rotate_list.append(filename)

# insert image to background and save

img_index = 0

for i in range(len(background_list)):
    background = cv2.imread(background_dir + '/' + background_list[i])
    # random resize background image
    scale = np.random.uniform(1.2, 1.5)
    background = cv2.resize(background, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    background_rotate = cv2.imread(background_rotate_dir + '/' + background_rotate_list[i])
    background_rotate = cv2.resize(background_rotate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    for j in range(len(original_list)):
        # insert to center of background
        original = cv2.imread(original_dir + '/' + original_list[j])
        scale = np.random.uniform(0.6, 1.1)

        original = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        

        h, w, c = original.shape
        
        bg_x, bg_y, bg_c = background.shape

        for idx in range(4):

            # random position so that the image is not out of the background
            x = np.random.randint(30, bg_x - h -40)
            y = np.random.randint(30, bg_y - w-40)

            # insert image to background
            copy = background.copy()
            copy[x:x+h, y:y+w] = original

            # save image
            cv2.imwrite('result/' + str(background_list[i])+ '/' + str(img_index) + '.jpg', copy)
            img_index += 1

        
        # rotate background image 10 degree


        for _ in range(4):

                bg_x, bg_y, bg_c = background_rotate.shape

                x = bg_x//2 - 500
                y = bg_y//2 - 500

                # insert image to background
                copy = background_rotate.copy()

                copy[x:x+h, y:y+w] = original

                # rotate background image 10 degree
                de = np.random.randint(1, 20)

                M = cv2.getRotationMatrix2D((x, y), de, 1)

                copy = cv2.warpAffine(copy, M, (bg_x, bg_y))

                a = np.random.randint(x - 180, x - 110)
                b = np.random.randint(y - 140, y - 60)

                # cut image to random size containing the original image

                ran_x = np.random.randint(700, 900)
                ran_y = np.random.randint(750, 1000)

                copy = copy[a:a+ran_y, b:b+ran_x]

                # save image

                cv2.imwrite('result/' + str(background_list[i])+ '/' + str(img_index) + '.jpg', copy)

                img_index += 1

                # rotate background image -10 degree
                de = np.random.randint(-20, 0)

                M = cv2.getRotationMatrix2D((x, y),de , 1)

                copy = background_rotate.copy()

                copy = cv2.warpAffine(copy, M, (bg_x, bg_y))

                copy[x:x+h, y:y+w] = original

                M = cv2.getRotationMatrix2D((x, y), -10, 1)

                copy = cv2.warpAffine(copy, M, (bg_x, bg_y))

                a = np.random.randint(x - 100, x - 60)
                b = np.random.randint(y - 180, y - 110)

                # cut image to 700x500 containing the original image
                ran_x = np.random.randint(700, 900)
                ran_y = np.random.randint(750, 1000)


                copy = copy[a:a+ran_y, b:b+ran_x]

                # save image

                cv2.imwrite('result/' + str(background_list[i])+ '/' + str(img_index) + '.jpg', copy)

                img_index += 1

    





        








