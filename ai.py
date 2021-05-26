from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
from PIL import Image
from file import File


class AI(object):
    def __init__(self):
        super(AI, self).__init__()
        # img
        self.file = File()
        self.img_combine_hd_arr = self.file.covert_image('model/0519_crop_hd_save_pics/')
        self.img_combine_sho_arr = self.file.covert_image('model/0519_crop_sho_save_pics/')
        self.img_combine_ft_arr = self.file.covert_image('model/0519_crop_ft_save_pics/')
        # ans
        self.ans = pd.read_csv('model/0519_wrong_total_sit_value_combinefoot_0_9000.csv')
        # ans
        self.ans_hd_arr = np.array(self.ans['head'])
        self.ans_sho_arr = np.array(self.ans['shoulder'])
        self.ans_ft_arr = np.array(self.ans['foot'])
        self.hd_model = load_model('model/crop_hd_cnn.h5')
        self.sho_model = load_model('model/crop_sho_cnn.h5')
        self.ft_model = load_model('model/crop_ft_cnn.h5')
        print('Model Reloaded')

    def init_default(self):
        # head part
        hd_x_train, hd_x_test, hd_y_train, hd_y_test = train_test_split(
            self.img_combine_hd_arr,
            self.ans_hd_arr,
            test_size=0.1
        )
        hd_x_train_norm, hd_x_test_norm = hd_x_train / 255, hd_x_test / 255
        # shoulder part
        sho_x_train, sho_x_test, sho_y_train, sho_y_test = train_test_split(
            self.img_combine_sho_arr,
            self.ans_sho_arr,
            test_size=0.1
        )
        sho_x_train_norm, sho_x_test_norm = sho_x_train / 255, sho_x_test / 255
        # foot part
        ft_x_train, ft_x_test, ft_y_train, ft_y_test = train_test_split(
            self.img_combine_ft_arr,
            self.ans_ft_arr,
            test_size=0.1)
        ft_x_train_norm, ft_x_test_norm = ft_x_train / 255, ft_x_test / 255
        # create model
        layers = [
            Conv2D(32, 3, padding="same", activation="relu", input_shape=(256, 256, 3)),
            MaxPooling2D(),
            Conv2D(64, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Conv2D(128, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Conv2D(256, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Conv2D(512, 3, padding="same", activation="relu"),
            GlobalAveragePooling2D(),
            Dense(1, activation="sigmoid")
        ]
        hd_model = Sequential(layers)
        # hd_model.summary()
        sho_model = Sequential(layers)
        # sho_model.summary()
        ft_model = Sequential(layers)
        # ft_model.summary()

        # compile
        hd_model.compile(loss=BinaryCrossentropy(), optimizer="adam", metrics=["accuracy"])
        sho_model.compile(loss=BinaryCrossentropy(), optimizer="adam", metrics=["accuracy"])
        ft_model.compile(loss=BinaryCrossentropy(), optimizer="adam", metrics=["accuracy"])

        # fit
        hd_model.fit(
            hd_x_train_norm, hd_y_train, batch_size=20, epochs=50, validation_split=0.1, verbose=2,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint("crop_hd_cnn.h5", save_best_only=True)
            ])
        sho_model.fit(
            sho_x_train_norm, sho_y_train, batch_size=20, epochs=50, validation_split=0.1, verbose=2,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint("crop_sho_cnn.h5", save_best_only=True)
            ])
        ft_model.fit(
            ft_x_train_norm, ft_y_train, batch_size=20, epochs=50, validation_split=0.1, verbose=2,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint("crop_ft_cnn.h5", save_best_only=True)
            ])

    def head_predict(self, file_path):
        im = Image.open(file_path).resize((256, 256)).convert('RGB')
        im = np.array(im)
        im = im.reshape(1, 256, 256, 3) / 255.0
        pre = self.hd_model.predict(im)[0][0]
        print('頭部預測數值', pre)
        return pre

    def shoulder_predict(self, file_path):
        im = Image.open(file_path).resize((256, 256)).convert('RGB')
        im = np.array(im)
        im = im.reshape(1, 256, 256, 3) / 255.0
        pre = self.sho_model.predict(im)[0][0]
        print('肩膀預測數值', pre)
        return pre

    def foot_predict(self, file_path):
        im = Image.open(file_path).resize((256, 256)).convert('RGB')
        im = np.array(im)
        im = im.reshape(1, 256, 256, 3) / 255.0
        pre = self.ft_model.predict(im)[0][0]
        print('腳部預測數值', pre)
        return pre

