from Face_Mesh import FaceMesh
import cv2
from torch.nn import functional as F
from torch import tensor, stack
import os
import numpy as np
from collections import Counter

def mark():
    class __Embeddings:
        def __init__(self):
            self.__targets = []
            vectors = []
            root_dir = 'Dataset'
            for name in os.listdir(root_dir):
                for orientation in os.listdir(os.path.join(root_dir, name)):
                    loc = os.path.join(root_dir, name, orientation)
                    self.__targets.append(f'{name}_{orientation.split("-")[-1]}')
                    vectors.append(tensor(np.load(loc)))

            self.__vectors = stack(vectors, dim=0)
            
        def similarity(self, mesh:list[np.array], x:int, y:int) -> str:
            preds = []
            for curr_mesh in mesh:
                curr_mesh = tensor(curr_mesh)
                sim = F.cosine_similarity(curr_mesh, self.__vectors)
                preds.append(self.__targets[sim.argmax()])
            return preds


    embeds = __Embeddings()
    face_mesh = FaceMesh(0.5, 0.5)
    cap = cv2.VideoCapture(0)

    mssg = ''
    cropped_image = None
    cap_imgs = []
    read_imgs = 0

    while cap.isOpened():
        _, image = cap.read()
        try:
            res_dict, cropped_image = face_mesh.parse_img(image)
            x, y, _ = res_dict['Orientation']
            cap_imgs.append(res_dict['Landmarks'])
            read_imgs += 1
            if read_imgs == 30:
                print(embeds.similarity(res_dict['Landmarks'], x, y))
                break
            x, y = round(x, 2), round(y, 2)
        except AssertionError:
            cap_imgs = []
            read_imgs = 0
            mssg = 'No Face Detected'
            
        cv2.putText(image, mssg, (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        cv2.imshow('Head Pose Detection',image)
        if cropped_image is not None:
            cv2.imshow('Detected Face', cropped_image)
        if cv2.waitKey(5) & 0xFF ==27:
            break

    cap.release()