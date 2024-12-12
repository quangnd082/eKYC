from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np 


def extrax_face(img):

    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device='cuda:0')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob > 0.9:
        face = face.cuda()
        img_embedding = resnet(face.unsqueeze(0))
        img_embedding = img_embedding.detach().cpu().numpy()
        return img_embedding
    else:
        return None
    
def compare_face(img1, img2):
    img1 = extrax_face(img1)
    img2 = extrax_face(img2)
    if img1 is not None and img2 is not None:
        dist = np.linalg.norm(img1 - img2)
        return dist
    else:
        return None
    
def compare_face2(img1, img2):
    img1 = extrax_face(img1)
    img2 = extrax_face(img2)
    if img1 is not None and img2 is not None:
        dist = np.linalg.norm(img1 - img2)
        if dist < 0.8:
            return True
        else:
            return False
    else:
        return False
    

def face_verify(img1,img2):
    isOne = compare_face2(img1,img2)
    return isOne
