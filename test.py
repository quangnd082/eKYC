import cv2
import onnxruntime as ort
from PIL import Image
import numpy as np
from vietocr.tool.predictor import Predictor   
from vietocr.tool.config import Cfg
import warnings 
warnings.filterwarnings("ignore")
import rotate



providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model/best.onnx", providers=providers)

config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = './transformerocr.pth'
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
config['cnn']['pretrained']=False

detector = Predictor(config)


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

img = cv2.imread('17.jpg')
# colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

img = rotate.main(img)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

image = img.copy()
image, ratio, dwdh = letterbox(image, auto=False)
image = image.transpose((2, 0, 1))
image = np.expand_dims(image, 0)
image = np.ascontiguousarray(image)

im = image.astype(np.float32)
im /= 255

outname = [i.name for i in session.get_outputs()]

inname = [i.name for i in session.get_inputs()]

inp = {inname[0]:im}

outputs = session.run(outname, inp)[0]

ori_images = [img.copy()]


outputs = np.array(outputs)

# select boxes with score >= 0.6

outputs = outputs[outputs[:,6]>=0.52]

# sort boxes by cls_id

outputs = outputs[outputs[:,5].argsort()]

id = []
name = []
sex = []
birth = []
nation = []
address1 = []
address2 = []


for i in outputs:
    if i[5] == 15:
        id.append(i)
    elif i[5] == 16:
        name.append(i)
    elif i[5]  == 18:
        sex.append(i)
    elif i[5] == 17:
        birth.append(i)
    elif i[5] == 19:
        nation.append(i)
    elif i[5] == 20:
        address1.append(i)
    elif i[5] == 21:
        address2.append(i)


# function predict

def predict(field):

    # print(field)
    # check if id field is 21
    if field[0][5] == 21:
        field = np.array(field)
        y_average =  (min(field[:,2]) + max(field[:,2])) / 2

        # sort field 21 by y0
        field = sorted(field, key=lambda x: ( x[2]))

        # split field 21 to 2 line
        field1 = []
        field2 = []

        for i in field:
            if i[2] <= y_average:
                field1.append(i)
            else:
                field2.append(i)

        # sort field 21 line 1 by x0
        field1 = sorted(field1, key=lambda x: ( x[1]))
        # sort field 21 line 2 by x0
        field2 = sorted(field2, key=lambda x: ( x[1]))

        text1 = ''
        for i in field1:
            image = ori_images[int(i[0])]
            box = np.array([i[1],i[2],i[3],i[4]])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            # crop image with expand 5 pixel
            crop_img = img[box[1]-5:box[3]+5, box[0]-5:box[2]+5]
            # format image to predict
            img_pil = Image.fromarray(crop_img)
            # predict
            text1 += detector.predict(img_pil) + " "

        # predict field 21 line 2
        text2 = ''
        for i in field2:
            image = ori_images[int(i[0])]
            box = np.array([i[1],i[2],i[3],i[4]])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            # crop image with expand 5 pixel
            crop_img = img[box[1]-5:box[3]+5, box[0]-5:box[2]+5]
            # format image to predict
            img_pil = Image.fromarray(crop_img)
            # predict
            text2 += detector.predict(img_pil) + " "



        return text1 + text2

    # sort field by y0 and x0
    field = sorted(field, key=lambda x: ( x[1]))

    text = ''
    for i in field:
        image = ori_images[int(i[0])]
        box = np.array([i[1],i[2],i[3],i[4]])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        # crop image with expand 5 pixel
        crop_img = img[box[1]-7:box[3]+7, box[0]-7:box[2]+7]
        # format image to predict
        img_pil = Image.fromarray(crop_img)
        # predict
        text += detector.predict(img_pil) + " "

    return text

try:

    print('id: ', predict(id))
    print('name: ', predict(name))
    print('sex: ', predict(sex))
    print('birth: ', predict(birth))
    print('nation: ', predict(nation))
    print('address1: ', predict(address1))
    print('address2: ', predict(address2))

except:
    print('error')
        



# print(outputs)

# for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
#     image = ori_images[int(batch_id)]
#     box = np.array([x0,y0,x1,y1])
#     box -= np.array(dwdh*2)
#     box /= ratio
#     box = box.round().astype(np.int32).tolist()
#     # crop image with expand 5 pixel
#     crop_img = img[box[1]-5:box[3]+5, box[0]-5:box[2]+5]
#     # format image to predict
#     img_pil = Image.fromarray(crop_img)
#     # predict
#     text = detector.predict(img_pil)
#     print(text)
