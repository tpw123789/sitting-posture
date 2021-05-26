import codecs
import json
import pandas as pd
import numpy as np
from PIL import Image


class CropImg:
    def __init__(self):
        # 關鍵點
        self.columns = {
            'coco_col': [
                'Nose_x', 'Nose_y', 'Neck_x', 'Neck_y',
                'R-Sho_x', 'R-Sho_y', 'R-Elb_x', 'R-Elb_y',
                'R-Wr_x', 'R-Wr_y', 'L-Sho_x', 'L-Sho_y',
                'L-Elb_x', 'L-Elb_y', 'L-Wr_x', 'L-Wr_y',
                'R-Hip_x', 'R-Hip_y', 'R-Knee_x', 'R-Knee_y',
                'R-Ank_x', 'R-Ank_y', 'L-Hip_x', 'L-Hip_y',
                'L-Knee_x', 'L-Knee_y', 'L-Ank_x', 'L-Ank_y',
                'R-Eye_x', 'R-Eye_y', 'L-Eye_x', 'L-Eye_y',
                'R-Ear_x', 'R-Ear_y', 'L-Ear_x', 'L-Ear_y'
            ],
            'shoulder': {
                'x': ["R-Sho_x", "L-Sho_x"],
                'y': ["R-Sho_y", "L-Sho_y"]
            },
            'head': {
                'x': ["Nose_x", "R-Eye_x", "R-Ear_x", "L-Eye_x", "L-Ear_x"],
                'y': ["Nose_y", "R-Eye_y", "R-Ear_y", "L-Eye_y", "L-Ear_y"]
            },
            'foot': {
                'x': ["R-Hip_x", "R-Knee_x", "R-Ank_x", "L-Hip_x", "L-Knee_x", "L-Ank_x"],
                'y': ["R-Hip_y", "R-Knee_y", "R-Ank_y", "L-Hip_y", "L-Knee_y", "L-Ank_y"]
            }
        }

        # 讀取json檔 -> 關鍵點DataFrame
        with codecs.open('./media/user_sent_key_points.json', mode='r', encoding='utf-8') as obj_text:
            self.data = json.loads(obj_text.read())
            self.data = np.array(self.data).reshape(1, 36)
            self.key_points = pd.DataFrame(self.data, columns=self.columns["coco_col"])

    def body_crop(self, file_path, body_part):
        x = self.key_points[self.columns[body_part]['x']][0:1]
        x = x.astype(int).iloc[0].values
        idx = np.nonzero(x)
        left, right = (0, 10) if sum(x) == 0 else (min(x[idx]) - 10, max(x[idx]) + 10)
        y = self.key_points[self.columns[body_part]['y']][0:1]
        y = y.astype(int).iloc[0].values
        idx = np.nonzero(y)
        top, bottom = (0, 10) if sum(y) == 0 else (min(y[idx]) - 10, max(y[idx]) + 10)
        with Image.open(file_path) as img:
            img_crop = img.crop((left, top, right, bottom))
            img_path = './media/crop_' + body_part + '.jpg'
            img_crop.save(img_path)
            img_crop = np.array(img_crop)
        print('Successfully crop ' + body_part + ' image.')
        return img_path






