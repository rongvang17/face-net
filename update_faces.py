import glob
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import shutil
import argparse

from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image

# ALL_IMG_PATH = 'data/test_images/face_database/'
# ADD_IMG_PATH = 'data2/test_images2/'
# faces_list path
save_data_path = '/home/minhthanh/directory_env/my_env/faceNet-Infer/'

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
    # pretrained="vggface2"
).to(device)
model.eval()

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform(img)
    
def create_embeddings(ALL_IMG_PATH, ADD_IMG_PATH):
    embeddings = []
    names = []
    requirement_add_infor = False

    for usr in os.listdir(ADD_IMG_PATH):
        check_exist = False
        for all_usr in os.listdir(ALL_IMG_PATH):
            if usr == all_usr:
                print("{} already exists in DB".format(usr))
                check_exist = True
                break
        if not check_exist:
            temp_img_path = os.path.join(ALL_IMG_PATH, usr)
            usr_path = os.path.join(ADD_IMG_PATH, usr)
            print(usr_path)
            shutil.copytree(usr_path, temp_img_path)

            embeds = []
            name_images = os.listdir(usr_path)
            print(names)
            for name_image in name_images:
                requirement_add_infor = True
                image_path = os.path.join(usr_path, name_image)
                img = cv2.imread(image_path)
                img = cv2.resize(img,(160, 160), interpolation=cv2.INTER_AREA)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    embeds.append(model(trans(img).to(device).unsqueeze(0))) # 1 anh, kich thuoc [1,512]

            if len(embeds) == 0:
                continue
            embedding = torch.cat(embeds) 
            embeddings.append(embedding) # 1 cai list n cai [1,512]
            names.append([usr] * embedding.shape[0])
        else:
            continue
    
    return embeddings, names, requirement_add_infor

def save_embeddings(embeddings, names):
    embeddings = torch.cat(embeddings) #[n,512]

    names_flat = []
    for sublist in names:
        names_flat.extend(sublist)
    names = np.array(names_flat).reshape(-1)

    # save embedding to .csv
    embeddings_csv = 'embeddings.csv'
    embeddings_path = os.path.join(save_data_path, embeddings_csv)
    if os.path.exists(embeddings_path):
        embeddings_np = embeddings.cpu().numpy().astype(float)
        df = pd.DataFrame(embeddings_np)
        df.to_csv(embeddings_path, mode='a', header=False, index=False)
    else:
        embeddings_np = embeddings.cpu().numpy().astype(float)
        df = pd.DataFrame(embeddings_np)
        df.to_csv(embeddings_path, index=False)

    # save names to .csv
    names_csv = 'names.csv'
    names_path = os.path.join(save_data_path, names_csv)
    if os.path.exists(names_path):
        names_series = pd.Series(names)
        names_df = pd.DataFrame(names_series, columns=['Name'])
        names_df.to_csv(names_path, mode='a', header=False, index=False)
    else:
        os.mkdir(names_path)
        names_series = pd.Series(names)
        names_df = pd.DataFrame(names_series, columns=['Name'])
        names_df.to_csv(names_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='get all_infor_path')
    parser.add_argument('all_img_path', help='all_faces_list')
    parser.add_argument('add_img_path', help='names_need_add')

    args = parser.parse_args()

    ALL_IMG_PATH = args.all_img_path
    ADD_IMG_PATH = args.add_img_path

    embeddings, names, requirement_add_infor = create_embeddings(ALL_IMG_PATH, ADD_IMG_PATH)
    if requirement_add_infor:
        save_embeddings(embeddings, names)


if __name__ == '__main__':
    main()
