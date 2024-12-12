from flask import Flask, render_template, request, jsonify, Response
from flask_session import Session
import cv2
import numpy as np
from io import BytesIO
import base64
from PIL import Image
#from verify import face_verify
import handle
import rotate


app = Flask(__name__)
app.config["SECRET_KEY"] = "OCML3BRawWEUeaxcuKHLpw"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)



@app.route('/')
def index():
    image_ID = None
    return render_template('index.html')

@app.route('/' , methods=['POST'])
def index_post():
    global image_ID
    if request.method == 'POST':

        image = request.files['file']


        # convert image to cv2 image
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        image_ID = image.copy()
        image = rotate.main(image)
        # use handle to predict
        id,name, birth, sex, nation, address1, address2 = handle.main(image, image_ID)
        # send data to index.html to display it 

        return jsonify({
            "id": id,
            "name": name,
            "birth": birth,
            "sex": sex,
            "nation": nation,
            "address1": address1,
            "address2": address2
        })
    

# @app.route('/verify', methods=['POST'])
# def check():
#     if image_ID is None:
#         return render_template('index.html')
#     return render_template('verify.html')

# @app.route('/face_verify', methods=['POST'])
# def _verify():
#     """        body: JSON.stringify({
#           img1: canvas1.toDataURL('image/png'),
#           img2: canvas2.toDataURL('image/png'),
#           img3: canvas3.toDataURL('image/png')
#         }),
#         """
#     # get data from request
#     data = request.get_json()

#     image_list = data['images']

#     for i, image_data in enumerate(image_list):
#         # Chuyển đổi dữ liệu Base64 thành đối tượng ảnh
#         image_data = image_data.split(",")[1]
#         image = Image.open(BytesIO(base64.b64decode(image_data)))
#         #  convert to cv2 format
#         image = np.array(image)
#         image_list[i] = image

#     # check if the same person
#     count = 0
#     for i in range(len(image_list)):
#         if face_verify(image_ID, image_list[i]):
#             count += 1
#     if count >= 2:
#         count = True
#     else:
#         count = False

#     print(count)
#     return jsonify({
#         "result": count
#     })


    

if __name__ == '__main__':
    app.run(debug=True)
