import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# read image from folder
# NAME_PATH = './data/test_images/face_database/'
# save_img_path = './data2/test_images2/'

# mtcnn = MTCNN(margin = 20, keep_all=True, select_largest = False, post_process=False, device = device)

# for name in os.listdir(NAME_PATH):
#     data_path = os.path.join(NAME_PATH, name)

#     name_images = os.listdir(data_path)
#     for name_image in name_images:
#         image_path = os.path.join(data_path, name_image)
#         frame = cv2.imread(image_path)
#         path = str(os.path.join(save_img_path, name) + '/{}.jpg'.format(os.path.splitext(name_image)[0]))
#         face_img = mtcnn(frame, save_path = path)
# read image from camera

def extract_face(box, img):
    face_size = 160
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    return face

IMG_PATH = '/home/minhthanh/directory_env/my_env/faceNet-Infer/data2/test_images2'
count = 8
usr_name = input("Input ur name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
if not os.path.exists(USR_PATH):
    os.mkdir(USR_PATH)

leap = 1

mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8] ,keep_all=True, device=device)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while cap.isOpened() and count:
    isSuccess, frame = cap.read()
    if isSuccess and leap%2==0:
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int, box.tolist()))
                face = extract_face(bbox, frame)
                cv2.imshow('img', face)
                cv2.waitKey(0)
                print(USR_PATH)
                cv2.imwrite(os.path.join(USR_PATH, 'viet_tan_{}.jpg'.format(count)), face)
    leap += 1
    count -= 1
    cv2.imshow('Face Capturing', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

