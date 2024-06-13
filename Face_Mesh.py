import numpy as np
import cv2
import mediapipe as mp
import time
from torchvision.datasets import ImageFolder

__vectorized_dict = {}

class ImageDataset(ImageFolder):
    def __init__(self, root='Image_Dataset'):
        super().__init__(root=root)

    def __len__(self):
        len(self.imgs)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        return path, target

class FaceMesh():
    def __init__(self, min_detect, min_track):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.__face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=min_detect, min_tracking_confidence=min_track)
        self.__face_crop = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    @classmethod
    def __create_landmarks(cls, results, img_shape):
        img_h , img_w, img_c = img_shape
        face_2d = []
        face_3d = []

        landmark_array = np.zeros((468, 3), dtype=float)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                        if idx ==1:
                            nose_2d = (lm.x * img_w,lm.y * img_h)
                            nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                        x,y = int(lm.x * img_w),int(lm.y * img_h)
                        face_2d.append([x,y])
                        face_3d.append(([x,y,lm.z]))
                    landmark_array[idx] = [lm.x, lm.y, lm.z]

                #Get 2d Coord
                face_2d = np.array(face_2d,dtype=np.float64)

                face_3d = np.array(face_3d,dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length,0,img_h/2],
                                    [0,focal_length,img_w/2],
                                    [0,0,1]])
                distortion_matrix = np.zeros((4,1),dtype=np.float64)

                _,rotation_vec,_ = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


                #getting rotational of face
                rmat,_ = cv2.Rodrigues(rotation_vec)

                (x,y,z),_,_,_,_,_ = cv2.RQDecomp3x3(rmat)

                return x * 360, y * 360, z * 360, landmark_array

    def parse_img(self, image):
        # Crop image
        face = self.__face_crop.detectMultiScale(image)
        
        assert len(face) >= 1
        x = face[0][0]
        y = face[0][1]
        w = face[0][2]
        h = face[0][3]
        crop_img = image[y:y+h, x:x+w]
        image = cv2.resize(crop_img, (200, 200))

        image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB) #flipped for selfie view
        image.flags.writeable = False
        results = self.__face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        try:
            x, y, z, landmark_array = self.__create_landmarks(results, image.shape)
        except TypeError:
            raise AssertionError
        return {'Orientation': (x, y, z), 'Landmarks': landmark_array.flatten()}, image

def store_write_vector_coordinates(vector_dict:dict, path):
    def resolve_orientation(orientation_list:list):
        pass

    # 1st index stores the rotation vector and the rest store landmarks
    resolved_orientations = resolve_orientation(list([res_dict['Orientation']]))
    ret_list = list(map(lambda row: (row.x, row.y, row.z), res_dict['Landmarks'][0].landmark))
    ret_arr = np.array(ret_list)
    if type(path) == type(''):
        np.save(file=path, arr=ret_list, allow_pickle=True)
    else:
        try:
            __vectorized_dict[path].append(ret_arr)
        except KeyError:
            __vectorized_dict[path] = [ret_arr]

    return ret_arr

if __name__ == '__main__':
    missed_dataset = {}
    img_dataset = ImageDataset()
    start = time.time()
    net_orientation = FaceMesh(0.5, 0.5)
    for img_path, label in img_dataset:
        img = cv2.imread(img_path)
        try:
            res_dict, cropped_image = net_orientation.parse_img(img)
            store_write_vector_coordinates(res_dict, img_path)
        except AssertionError:
            print('No Face/Map Detected')
            try:
                missed_dataset[img_path] += 1
            except KeyError:
                missed_dataset[img_path] = 1
            continue

    end = time.time()

    print(f'time taken: {end - start}')
    print('\nMissing Records\n\t')
    print(*missed_dataset, sep='\n\t')

    cv2.destroyAllWindows()