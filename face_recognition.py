import cv2
import torch
import numpy as np
import time
import pandas as pd

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
# from retinaface import RetinaFace


frame_size = (640,480)
IMG_PATH = './data/test_images'
DATA_PATH = './data'

# convert image to tensor
def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor()
            # fixed_image_standardization
        ])
    return transform(img)

# load faces_list
def load_faceslist(embeddings_path, names_path):
    # read embeddings
    embeddings_df = pd.read_csv('embeddings.csv')
    embeddings_np = embeddings_df.to_numpy()
    embeddings = torch.tensor(embeddings_np, dtype=torch.float)

    # read names
    names_df = pd.read_csv('names.csv')
    names_np = names_df['Name'].to_numpy()
    names = names_np.tolist()
    # print(names)

    return embeddings, names

# check face matching
def inference(model, face, local_embeds):
    embeds = []

    with torch.no_grad():
        embeds.append(model(trans(face).to(device).unsqueeze(0)))
        detect_embeds = torch.cat(embeds) #[1,512]
   
    cosin_arr = []
    word1_embedding = detect_embeds.detach().numpy().reshape(1, -1)

    for i in range(local_embeds.size(0)):
        word2_embedding = local_embeds[i].detach().numpy().reshape(1, -1)
        similarity = cosine_similarity(word1_embedding, word2_embedding)[0][0]
        cosin_arr.append(similarity)

    index_max_cosin = cosin_arr.index(max(cosin_arr))

    if cosin_arr[index_max_cosin] > 0.7:      # matching
        print(cosin_arr[index_max_cosin])
        return index_max_cosin, cosin_arr[index_max_cosin]
    else:
        return -1, 0
    
# get box face
def extract_face(box, img):
    face_size = 160
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    return face


if __name__ == "__main__":
    prev_frame_time = 0
    new_frame_time = 0

    # checking device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # recognition module
    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
    model.eval()

    # detection module
    mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8] ,keep_all=True, device=device)

    # set frame size
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    # load faces_list
    embeddings_path = 'embeddings.csv'
    names_path = 'names.csv'
    embeddings, names = load_faceslist(embeddings_path, names_path)

    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = extract_face(bbox, frame)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    idx, score = inference(model, face, embeddings)
                    if idx != -1:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                    else:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1)&0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# resp = RetinaFace.detect_faces(frame)
# for facial_key, facial_value in resp.items():

#     facial_area = facial_value['facial_area']
#     face = extract_face(facial_area, frame)
#     idx, score = inference(model, face, embeddings)
#     if idx != -1:
#         frame = cv2.rectangle(frame, (facial_area[0],facial_area[1]), (facial_area[2],facial_area[3]), (0,0,255), 6)
#         score = torch.Tensor.cpu(score[0]).detach().numpy()*power
#         frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (facial_area[0],facial_area[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
#     else:
#         frame = cv2.rectangle(frame, (facial_area[0],facial_area[1]), (facial_area[2],facial_area[3]), (0,0,255), 6)
#         frame = cv2.putText(frame,'Unknown', (facial_area[0],facial_area[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)

