from PIL import Image
import glob
import cv2
from io import BytesIO
import numpy as np


class File(object):
    def __init__(self):
        super(File, self).__init__()

    def save_bytes_image(self, raw):
        file_path = './media/user_sent.jpg'
        img = Image.open(BytesIO(raw))
        img.save(file_path)
        return img, file_path

    def image_name(self, file_folder):
        fn_list = []
        paths = sorted(glob.glob(file_folder + "*"))
        for path in paths:
            fn = path.split('\\')[-1].split('.')[0]
            fn_list.append(fn)
        return fn_list

    def covert_image(self, file_folder):
        img_arr_list = []
        fn_list = self.image_name(file_folder)
        for fn in fn_list:
            img = Image.open(file_folder + fn + ".jpg").convert("RGB")
            img = img.resize((256, 256))
            globals()["img_" + fn] = img
            globals()["img_" + fn + "_arr"] = np.array(globals()["img_"+fn])
            img_arr_list.append(globals()["img_" + fn + "_arr"])
            # img_arr_list = ()
        img_combine_arr = np.array(img_arr_list)
        return img_combine_arr



