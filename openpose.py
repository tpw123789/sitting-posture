import sys
import os
from sys import platform
import argparse
import cv2

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
    """ Custom Params (refer to include/openpose/flags.hpp for more parameters) """
    def __init__(self):
        self.params = dict()
        self.params['model_folder'] = 'openpose/models/'
        self.params["net_resolution"] = "-1x160"
        self.params["model_pose"] = "COCO"
        self.imageToProcess = cv2.imread('./media/user_sent.jpg')
        self.imageToProcess = cv2.resize(self.imageToProcess, (256, 256), interpolation=cv2.INTER_CUBIC)

    def skeleton_image(self):
        self.params["disable_blending"] = 'True'
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(self.params)
        opWrapper.start()
        self.params["disable_blending"] = 'False'

        # Process Image
        datum = op.Datum()
        datum.cvInputData = self.imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        cv2.imwrite('./media/user_sent_skeleton.jpg', datum.cvOutputData)

    def people_skeleton_image(self):
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(self.params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        datum.cvInputData = self.imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        cv2.imwrite('./media/user_sent_people_skeleton.jpg', datum.cvOutputData)

a = OpenPose()
a.skeleton_image()
a.people_skeleton_image()


# # Save Image
# print('Body key points: ')
# a = datum.poseKeypoints
# cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)




