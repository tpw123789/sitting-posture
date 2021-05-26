import sys
import os
from sys import platform
import cv2
import codecs
import json

# Import OpenPose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == 'win32':
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/openpose/build/python/openpose/Release')
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/openpose/build/x64/Release;' + dir_path + '/openpose/build/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('/openpose/python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu),
        # you can also access the OpenPose/python module from there.
        # This will install OpenPose and the python library at your desired installation path.
        # Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` '
          'in CMake and have this Python script in the right folder?')
    raise e


class OpenPose():
    def __init__(self, file_path):
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        self.params = dict()
        self.params['model_folder'] = 'openpose/models/'
        self.params["net_resolution"] = "-1x160"
        self.params["model_pose"] = "COCO"
        self.params["disable_blending"] = 'True'
        self.imageToProcess = cv2.imread(file_path)
        self.imageToProcess = cv2.resize(self.imageToProcess, (256, 256), interpolation=cv2.INTER_CUBIC)

    # Starting OpenPose
    @staticmethod
    def openpose_wrapper(params):
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        return opWrapper

    # Process Image
    def process_image(self, opWrapper):
        datum = op.Datum()
        datum.cvInputData = self.imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        # 回傳圖片陣列+關鍵點陣列
        return datum.cvOutputData, datum.poseKeypoints

    # 骨架圖
    def skeleton_image(self):
        opWrapper = self.openpose_wrapper(self.params)
        img_array, key_points = self.process_image(opWrapper)
        file_path = './media/user_sent_skeleton.jpg'
        cv2.imwrite(file_path, img_array)
        print('Body key points:\n{}'.format(key_points))
        with codecs.open('media/user_sent_key_points.json', 'w', encoding='utf-8') as fn:
            json.dump(key_points[0][:, [0, 1]].tolist(), fn, indent=4)
        return file_path

    # 原圖 + 骨架
    def people_skeleton_image(self):
        self.params["disable_blending"] = 'False'
        opWrapper = self.openpose_wrapper(self.params)
        self.params["disable_blending"] = 'True'
        img_array, key_points = self.process_image(opWrapper)
        file_path = './media/user_sent_people_skeleton.jpg'
        cv2.imwrite(file_path, img_array)
        print('Body key points:\n{}'.format(key_points))
        with codecs.open('media/user_sent_key_points.json', 'w', encoding='utf-8') as fn:
            json.dump(key_points[0][:, [0, 1]].tolist(), fn, indent=4)
        return file_path




