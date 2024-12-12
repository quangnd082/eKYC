import cv2
import onnxruntime as ort
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model/q_best.onnx", providers=providers)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)


    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def main(img):

    img = img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (640, 640))

    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    outname = [i.name for i in session.get_outputs()]
    outname

    inname = [i.name for i in session.get_inputs()]
    inname

    inp = {inname[0]:im}

    pred = session.run(outname, inp)[0]


    # get points from output

    pred = pred[pred[:,5].argsort()]

    pred=pred[pred[:,6]>0.35]

    try:


        top_left = pred[0][1:5]
        top_left -= np.array(dwdh*2)
        top_left /= ratio
        bot_left = pred[1][1:5]
        bot_left -= np.array(dwdh*2)
        bot_left /= ratio
        top_right = pred[2][1:5]
        top_right -= np.array(dwdh*2)
        top_right /= ratio
        bot_right = pred[3][1:5]
        bot_right -= np.array(dwdh*2)
        bot_right /= ratio


        top_left = top_left[0], top_left[1] 
        bot_left = bot_left[0], bot_left[3]
        top_right = top_right[2], top_right[1]
        bot_right = bot_right[2], bot_right[3]

    except:
        print("Can't rotate")
        return img



    # get max width and height

    # max_width = int(max(np.linalg.norm(np.array(top_left) - np.array(top_right)), np.linalg.norm(np.array(bot_left) - np.array(bot_right))))
    # max_height = int(max(np.linalg.norm(np.array(top_left) - np.array(bot_left)), np.linalg.norm(np.array(top_right) - np.array(bot_right))))

    max_width=800
    max_height=502

    input_points = np.float32([top_left, bot_left, bot_right, top_right])
    output_points = np.float32([[0, 0],
                            [0, max_height - 1],
                            [max_width - 1, max_height - 1],
                            [max_width - 1, 0]])

    matrix = cv2.getPerspectiveTransform(input_points, output_points)
    img_output = cv2.warpPerspective(img, matrix, (max_width, max_height))
    return img_output

# img = cv2.imread("11.jpg")
# img = main(img)
# # adujst gamma

# img = skimage.exposure.adjust_gamma(img, gamma=6, gain=1)
# plt.imshow(img)
# plt.show()
