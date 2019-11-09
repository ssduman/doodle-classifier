from PIL import Image, ImageGrab
import tkinter as tk
from tkinter import messagebox, ttk
import time
from NeuralNetwork import *
from GUI import *
from math import ceil
import os

class DoodleClassifier(object):
    def __init__(self):
        self.gui = GUI()
        self.root = self.gui.getRoot()

        self.canvas = tk.Canvas(self.root, width=300, height=300, bg="white")

        self.img = None
        self.img_original = None

        self.names = []
        self.all_names = {}
        self.num_data = 0
        self.num_instance = 1000
        self.learning_rate = 0.1
        self.epoch = 1
        self.hidden1_size = 64
        self.hidden2_size = 32
        self.hidden3_size = 32
        self.all_data = []
        self.layers = []
        self.NN = None
        self.selected_data = {}
        self.train_and_test = {}
        self.painted = []

        self.button_ok = tk.Button(self.root, text="ok!", command=self.destroy_and_create_nn)
        self.button_load = tk.Button(self.root, text="load!", command=self.load_nn)

        self.frame_r_b = tk.Frame(self.root)
        self.button_predict = tk.Button(text="predict", command=self.button_predict_f)
        self.button_save = tk.Button(text="save", command=lambda: self.img_original.save("draw.png"))
        self.button_clear = tk.Button(text="clear", command=self.button_clear_f)
        self.textArea = tk.Text(self.frame_r_b, font=("", 10), width=25)

        self.button_true = tk.Button(self.root, text="True!", command=self.predicted_true)
        self.button_save_nn = tk.Button(self.root, text="Save!", command=self.button_save_nn_f)
        self.button_false = tk.Button(self.root, text="False!", command=self.predicted_false)
        self.label_select_true = tk.Label(self.frame_r_b, text="select true")
        self.combo_box = tk.ttk.Combobox(self.frame_r_b, width=25)

        self.data_exist = False
        if os.path.isdir("data"):
            self.data_exist = True
            self.data_checkbutton, self.all_names, self.num_data = self.gui.create_check_buttons()

        self.gui.grid_select_area(self.data_exist, self.num_data)
        if self.data_exist:
            self.button_ok.pack(side=tk.BOTTOM)
            self.button_load.pack(side=tk.BOTTOM, pady=2)
        else:
            self.button_load.pack(side=tk.BOTTOM, pady=2)

        self.root.mainloop()

    def load_nn(self):
        self.names = np.loadtxt("names.txt", dtype=str)

        self.gui.destroy_select_area()

        self.button_ok.destroy()
        self.button_load.destroy()

        self.pack_draw_area()

        start = time.time()

        self.NN = NeuralNetwork(None, self.names, from_load=True)

        if self.data_exist:
            self.select_data()
            self.concatenate_data()
            self.calculate_accuracy()

        print("time taken: {0:.2f} sec".format(time.time() - start))

    def destroy_and_create_nn(self):
        marked = False
        user_layer = []
        valid_entry, message, self.num_instance, self.learning_rate, self.epoch, user_layer = self.gui.get_entries()

        for i in range(self.num_data):
            if self.data_checkbutton[i].get() == 1:
                marked = True
                self.names.append(self.all_names[i])

        if self.gui.data_checkbutton_errors(marked, valid_entry, message):
            self.names = []
            return

        self.gui.destroy_select_area()

        self.button_ok.destroy()
        self.button_load.destroy()

        self.pack_draw_area()

        start = time.time()

        self.select_data()
        self.concatenate_data()

        user_layer.insert(0, 28 * 28)
        user_layer.append(len(self.names))
        self.layers = user_layer
        self.NN = NeuralNetwork(self.layers, self.names, self.learning_rate, self.epoch)
        self.NN.train(self.all_data)
        self.calculate_accuracy()

        print("time taken: {0:.2f} sec".format(time.time() - start))

    def select_data(self):
        index = 0
        length = len(self.names)
        for name in self.names:
            target = [1 if index == x else 0 for x in range(length)]
            self.selected_data[index] = [name, target]
            train, test = self.prepare_data("data/full_numpy_bitmap_" + name + ".npy", target)
            self.train_and_test[index] = [train, test]
            index += 1

    def concatenate_data(self):
        self.all_data = np.array(self.train_and_test[0][0])
        length = len(self.train_and_test)

        for index in range(1, length):
            self.all_data = np.concatenate([self.all_data, self.train_and_test[index][0]], axis=0)

        np.random.shuffle(self.all_data)
        # self.draw10image(self.all_data)

    def prepare_data(self, name, y):
        data = np.load(name)
        data = data[:self.num_instance]

        data_training = np.array([(np.divide(d, 255), y) for d in data[:int(len(data) * 0.8)]])
        data_testing = np.array([(np.divide(d, 255), y) for d in data[len(data_training):len(data)]])

        return data_training, data_testing

    def calculate_accuracy(self):
        length = len(self.train_and_test)
        for index in range(length):
            acc = self.NN.accuracy(self.train_and_test[index][1])
            print("accuracy {}: %{}".format(self.selected_data[index][0], acc), end=", ")

    def draw_canvas(self, event):
        x1, y1 = (event.x - 0.5), (event.y - 0.5)
        x2, y2 = (event.x + 0.5), (event.y + 0.5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=8)

    def draw1image(self, data):
        img_data = np.full((28, 28, 3), 255, dtype=np.uint8)
        for i in range(28):
            for j in range(28):
                img_data[i, j] = data[i * 28 + j]

        img = Image.fromarray(img_data)
        img.show()

    def draw10image(self, data):
        img_data = np.full((280, 280, 2), 255, dtype=np.uint8)
        for pos in range(100):
            x_off = (pos // 10) * 28
            y_off = (pos % 10) * 28
            for i in range(28):
                for j in range(28):
                    a = data[pos][0][i * 28 + j]
                    x = int(x_off) + i
                    y = int(y_off) + j
                    img_data[x, y] = [255 - a * 255, 255 - a * 255]

        img = Image.fromarray(img_data)
        img.show()

    def take_ss(self):
        x1 = self.root.winfo_rootx() + (self.root.winfo_rootx() * 0.25) + 5
        y1 = self.root.winfo_rooty() + (self.root.winfo_rooty() * 0.25) + 5
        x2 = x1 + self.canvas.winfo_width() + (self.canvas.winfo_width() * 0.25) - 5
        y2 = y1 + self.canvas.winfo_height() + (self.canvas.winfo_height() * 0.25) - 5
        self.img_original = ImageGrab.grab().crop((x1, y1, x2, y2))
        self.img = self.img_original.resize((28, 28))

    def button_predict_f(self):
        self.take_ss()
        painted_to_draw = []

        for i in range(28):
            for j in range(28):
                data = list(self.img.getpixel((j, i)))
                brightness = ceil(0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2])
                brightness /= 255
                painted_to_draw.append(data)
                self.painted.append(1 - brightness)

        # draw1image(painted)

        predict_picture = self.NN.predict(self.painted)
        self.textArea.delete("1.0", tk.END)
        print("it is a " + self.names[predict_picture])
        self.textArea.insert("1.0", "it is a " + self.names[predict_picture])

        self.user_feedback()

    def user_feedback(self):
        self.button_true.pack()
        self.button_true.configure(bg="#289946", activebackground="#84b591", foreground="white")
        self.canvas.create_window(30, 280, anchor="sw", window=self.button_true)

        self.button_save_nn.pack()
        self.button_save_nn.configure(bg="#6f9de8", activebackground="#94b8f2", foreground="white")
        self.canvas.create_window(170, 280, anchor="se", window=self.button_save_nn)

        self.button_false.pack()
        self.button_false.configure(bg="#e64c70", activebackground="#e68198", foreground="white")
        self.canvas.create_window(270, 280, anchor="se", window=self.button_false)

    def predicted_true(self):
        self.button_true.pack_forget()
        self.button_false.pack_forget()
        self.button_clear_f()

    def button_save_nn_f(self):
        self.button_save_nn.pack_forget()
        self.NN.save()

    def predicted_false(self):
        self.button_true.pack_forget()
        self.button_false.pack_forget()

        self.textArea.grid_forget()

        self.label_select_true.grid(row=0, column=0, sticky="w", padx=5)

        true_one = tk.StringVar()
        values = [x for x in self.names]
        self.combo_box.configure(textvariable=true_one, values=values)
        self.combo_box.grid(row=1, column=0, sticky="w", padx=5)
        self.combo_box.bind("<<ComboboxSelected>>", self.combo_box_selected)

    def combo_box_selected(self, event):
        search = self.combo_box.get()
        target = None
        index = 0
        for name in self.names:
            if search == name:
                target = [1 if x == index else 0 for x in range(len(self.names))]
                break
            index += 1

        train_user_input = np.array([np.array(self.painted), target])
        self.NN.train(train_user_input, single=True)

        if self.data_exist:
            self.calculate_accuracy()

        self.combo_box.grid_forget()
        self.label_select_true.grid_forget()
        self.textArea.grid()

        self.button_clear_f()

    def button_clear_f(self):
        self.painted.clear()
        self.img = None
        self.img_original = None
        self.canvas.delete("all")
        self.textArea.delete("1.0", tk.END)
        self.textArea.insert("1.0", "draw a " + self.names.__str__()[1:-2] + "!")

    def pack_draw_area(self):
        self.canvas.pack(fill=tk.BOTH)

        self.button_predict.pack(fill=tk.BOTH, side=tk.LEFT, expand=1)
        self.button_save.pack(fill=tk.BOTH, side=tk.LEFT, expand=1)
        self.button_clear.pack(fill=tk.BOTH, side=tk.LEFT, expand=1)
        self.frame_r_b.pack(fill=tk.BOTH, side=tk.LEFT, expand=1)

        self.textArea.insert("1.0", "draw a " + self.names.__str__()[1:-2] + "!")
        self.textArea.grid()

        self.canvas.bind("<B1-Motion>", self.draw_canvas)


if __name__ == '__main__':
    DoodleClassifier()
