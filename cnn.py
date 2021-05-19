from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
from PIL import Image
from file import File

# img
img_folder = 'out_resize256x256_total_skeleton/'
file = File()
img_combine_arr = file.covert_image(img_folder)  # nparray
# ans csv
ans = pd.read_csv('wrong_total_sit_value_check.csv')
ans_arr = np.array(ans)


class AI(object):
    def __init__(self):
        super(AI, self).__init__()
        self.ans = ans
        self.ans_arr = ans_arr
        self.img_combine_arr = img_combine_arr
        self.init_default()

    def init_default(self):
        # split data
        x_train, x_test, y_train, y_test = train_test_split(img_combine_arr, ans_arr, test_size=0.1)
        x_train, x_test = x_train / 255, x_test / 255
        # model
        layers = [
            Conv2D(64, 3, padding="same", activation="relu", input_shape=(256, 256, 3)),
            MaxPooling2D(),
            Conv2D(128, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Conv2D(256, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Conv2D(512, 3, padding="same", activation="relu"),
            GlobalAveragePooling2D(),
            Dense(3, activation="sigmoid")
        ]
        self.model = Sequential(layers)
        # model.summary()
        self.model.compile(loss=BinaryCrossentropy(), optimizer="adam", metrics=["accuracy"])
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint("cnn.h5", save_best_only=True)
        ]
        # self.model.fit(x_train, y_train, batch_size=20, epochs=50, validation_split=0.1, verbose=2, callbacks=callbacks)
        self.model.load_weights('cnn.h5')
        print('Model Reloaded')

    def predict_image_with_path(self, file_path):
        im = Image.open(file_path).resize((256, 256)).convert('RGB')
        im = np.array(im)
        im = im.reshape(1, 256, 256, 3) / 255.0
        pre = self.model.predict(im)[0]
        print("預測數值", pre.tolist())
        ans_names = self.ans.columns.tolist()
        ans_list = [round(i) for i in pre.tolist()]
        ans_trans = []
        for i, ans_val in zip(ans_names, ans_list):
            ans_trans.append(i + '_wrong' if ans_val == 1 else i + '_correct')
        return '預測數值{},\n {}'.format(pre.tolist(), ans_trans)