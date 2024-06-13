from Face_Mesh import FaceMesh
import cv2
import time
import os
import numpy as np
from tqdm import tqdm

def load_data(root_loc:str):
    face_mesh = FaceMesh(0.5, 0.5)
    base_dir = 'Dataset'
    for name, _ in zip(os.listdir(root_loc), tqdm(range(len(os.listdir(root_loc))))):
        person_path = os.path.join(root_loc, name)
        for i, img in enumerate(os.listdir(person_path)):
            img_path = os.path.join(person_path, img)
            img_read = cv2.imread(img_path)
            try:
                res_set, _ = face_mesh.parse_img(img_read)
                person_path_save = os.path.join(base_dir, name)
                if not os.path.exists(person_path_save):
                    os.mkdir(person_path_save)
                np.save(f'{person_path_save}\\{name}-{i}', res_set['Landmarks'])
            except AssertionError:
                pass

def capture_face(name:str):

    def update_mssg_and_vector(cmd:list, res_set:dict) -> str:
        nonlocal record_vector, i, name
        mssg, x_min, x_max, y_min, y_max = cmd[i]

        x, y, z = res_set['Orientation']

        if (x_min <= x <= x_max) and (y_min <= y <= y_max):
            check_val = None
            while check_val is None:
                check_val = input('Save image like this? (Y/N): ')
                if check_val.upper() in ['Y', 'N']:
                    break
            if check_val.upper() == 'Y':
                record_vector[f'{name}-{mssg.split()[1][0]}'] = res_set['Landmarks']
                i += 1
                if i >= 4:
                    return 'Close', -1, -1
                mssg = cmd[i][0]

        return mssg, x, y

    record_vector = {}
    i = 0
    command_list = [
        ('Look Right', -5, 5, 40, 50), 
        ('Look Left', -5, 5, -50, -40),
        ('Look Down', -50, -40, -5, 5),
        ('Look Up', 40, 50, -5, 5)
    ]

    face_mesh = FaceMesh(0.5, 0.5)
    cap = cv2.VideoCapture(0)

    mssg = ''
    cropped_image = None

    while cap.isOpened():
        _, image = cap.read()

        start = time.time()
        try:
            res_dict, cropped_image = face_mesh.parse_img(image)
            mssg, x, y = update_mssg_and_vector(command_list, res_dict)
            x, y = round(x, 2), round(y, 2)
        except AssertionError:
            mssg = 'No Face Detected'
            x, y = 'Nill', 'Nill'
        if mssg == 'Close':
            break
        
        # height = int(cap.get(4)) 
        # width = int(cap.get(3))
        # cv2.line(image, (0, 0), (width, height), (0, 0, 255), 5) 
        # cv2.line(image, (width, 0), (0, height), (0, 0, 255), 5) 
        cv2.putText(image, mssg, (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.putText(image, f'X: {x}', (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
        cv2.putText(image, f'Y: {y}', (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)

        end = time.time()
        totalTime = end-start
        try:
            fps = 1/totalTime
        except ZeroDivisionError:
            fps = -1
            pass
        cv2.putText(image,f'FPS: {int(fps)}',(20,450),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
        cv2.imshow('Head Pose Detection',image)
        if cropped_image is not None:
            cv2.imshow('Detected Face', cropped_image)
        if cv2.waitKey(5) & 0xFF ==27:
            break
    cap.release()
    return record_vector
