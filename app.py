# app.py
import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
import os
from PIL import Image, ImageTk
import numpy as np
from sklearn.svm import LinearSVC

class App:
    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):
        self.window = window
        self.window_title = window_title
        self.counters = [1, 1]
        self.model = LinearSVC()
        self.auto_predict = False
        self.camera = Camera()
        self.init_gui()
        self.delay = 15
        self.update()
        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=640, height=480)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring("Classname One", "Enter the name of the first class:", parent=self.window)
        self.classname_two = simpledialog.askstring("Classname Two", "Enter the name of the second class:", parent=self.window)

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=self.train_model)
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        folder_path = f'images/{class_num}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        img_path = f'{folder_path}/frame{self.counters[class_num-1]}.jpg'
        cv.imwrite(img_path, cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        self.counters[class_num - 1] += 1

    def reset(self):
        for folder in ['images/1', 'images/2']:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        self.counters = [1, 1]
        self.model = LinearSVC()
        self.class_label.config(text="CLASS")

    def update(self):
        if self.auto_predict:
            self.predict()

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def train_model(self):
        img_list = []
        class_list = []

        for class_num in [1, 2]:
            folder_path = f'images/{class_num}'
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                img = cv.resize(img, (150, 150))
                img_list.append(img.flatten())
                class_list.append(class_num)

        img_array = np.array(img_list)
        class_array = np.array(class_list)

        self.model.fit(img_array, class_array)
        print("Model successfully trained!")

    def predict(self):
        ret, frame = self.camera.get_frame()

        if ret:
            img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            img = cv.resize(img, (150, 150))
            img_flatten = img.flatten()
            prediction = self.model.predict([img_flatten])

            if prediction == 1:
                self.class_label.config(text=self.classname_one)
            elif prediction == 2:
                self.class_label.config(text=self.classname_two)

class Camera:
    def __init__(self):
        self.camera = cv.VideoCapture(0)
        if not self.camera.isOpened():
            raise ValueError("Unable to open camera!")

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    def get_frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()

            if ret:
                return (ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return None

if __name__ == "__main__":
    app = App()
