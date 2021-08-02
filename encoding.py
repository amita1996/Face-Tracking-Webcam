import cv2
import face_recognition as fr
import glob
import pickle


def get_images():
    return glob.glob("data/*.jpg")

def store_encoding():
    images_list = get_images()
    encoding_list = []
    for image in images_list:
        loaded_image = fr.load_image_file(image)
        loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)
        encoding_list.append(fr.face_encodings(loaded_image)[0])
        file = open('encoding_list','wb')
        pickle.dump(encoding_list,file)

store_encoding()