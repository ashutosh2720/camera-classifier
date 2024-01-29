# model.py
from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv

class Model:
    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, img_array, class_array):
        self.model.fit(img_array, class_array)
        print("Model successfully trained!")

    def predict(self, img_flatten):
        return self.model.predict([img_flatten])[0]
