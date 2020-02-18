from PIL import Image, ImageGrab
import tkinter as tk
from tkinter import messagebox, ttk
import time
from NeuralNetwork import *
from GUI import *
from math import ceil, floor
import random
import os

class DoodleClassifier(object):
    def __init__(self):
        self.gui = GUI()
        self.root = self.gui.getRoot()

        self.canvas = tk.Canvas(self.root, width=300, height=300, bg="white")

        np.random.seed(1)

        self.img = None
        self.img_original = None

        self.names = []
        self.all_names = {}
        self.num_data = 0
        self.num_instance = 5000
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
        self.button_show = tk.Button(self.root, text="show!", command=self.show)

        self.frame_r_b = tk.Frame(self.root)
        self.button_predict = tk.Button(text="predict", command=self.button_predict_f)
        self.button_save = tk.Button(text="save", command=lambda : self.img_original.save("draw.png"))
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
            self.button_ok.pack(side=tk.TOP, padx=20)
            self.button_load.pack(side=tk.LEFT, padx=35, pady=5)
            self.button_show.pack(side=tk.RIGHT, padx=35, pady=5)
        else:
            self.button_load.pack(side=tk.TOP, pady=20)

        self.data_X = None
        self.data_Y = None

        self.test_X = None
        self.test_Y = None

        self.root.mainloop()

    def load_nn(self):
        self.names = np.loadtxt("names.txt", dtype=str)

        self.gui.destroy_select_area()

        self.button_ok.destroy()
        self.button_load.destroy()
        self.button_show.destroy()

        self.pack_draw_area()

        start = time.time()

        self.NN = NeuralNetwork(None, self.names, from_load=True)

        if self.data_exist:
            self.load_selected_data()
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
        self.button_show.destroy()

        self.pack_draw_area()

        start = time.time()

        self.num_instance = 5000
        self.load_selected_data()
        
        user_layer.insert(0, 28 * 28)
        user_layer.append(len(self.names))

        self.layers = user_layer
        self.NN = NeuralNetwork(self.layers, self.names, l_rate=0.1)
        self.NN.tf(self.data_X, self.data_Y, self.test_X, self.test_Y)
        print()
        self.NN.train(self.data_X.T, self.data_Y.T, self.test_X.T, self.test_Y.T, optimizer="L2")
        self.calculate_accuracy()

        print("time taken: {0:.2f} sec".format(time.time() - start))

    def show(self):
        first = True
        data = []

        for i in range(self.num_data):
            if self.data_checkbutton[i].get() == 1:
                first = False
                name = self.all_names[i]
                d = np.load("data/full_numpy_bitmap_" + name + ".npy")
                for x in range(100):
                    data.append(d[x] / 255.)

        if first:
            self.gui.popup_message("mark at least one!")
        else:
            random.shuffle(data)
            self.drawNimage(np.array(data), N=10)

    def load_selected_data(self):
        index = 0
        length = len(self.names)
        for name in self.names:
            target = [1 if index == x else 0 for x in range(length)]
            self.load_data("data/full_numpy_bitmap_" + name + ".npy", target)
            index += 1
        
        m = self.data_X.shape[0]

        indices = np.arange(m)
        np.random.shuffle(indices)
        self.data_X = self.data_X[indices]
        self.data_Y = self.data_Y[indices]

        self.test_X = self.data_X[0:int(m * 0.1)]
        self.data_X = self.data_X[int(m * 0.1):m]

        self.test_Y = self.data_Y[0:int(m * 0.1)]
        self.data_Y = self.data_Y[int(m * 0.1):m]

        # self.drawNimage(self.data_X, N=20)
        # self.drawNimage(self.test_X, N=15)

        print("train -> X:", self.data_X.shape, self.data_Y.shape)
        print(" test -> X:", self.test_X.shape, self.test_Y.shape)

    def load_data(self, name, y):
        data = np.load(name)
        data = data[:self.num_instance]
        if y[0] == 1:
            self.data_X = data / 255.
            self.data_Y = np.full((data.shape[0], len(y)), y)
        else:
            self.data_X = np.vstack((self.data_X, data / 255.))
            temp = np.full((data.shape[0], len(y)), y)
            self.data_Y = np.vstack((self.data_Y, temp))

    def calculate_accuracy(self):
        train_acc = self.NN.accuracy(self.data_X.T, self.data_Y.T)
        test_acc = self.NN.accuracy(self.test_X.T, self.test_Y.T)
        print("train acc: %{:.2f}, test acc: %{:.2f}".format(train_acc, test_acc))

    def draw_canvas(self, event):
        x1, y1 = (event.x - 0.5), (event.y - 0.5)
        x2, y2 = (event.x + 0.5), (event.y + 0.5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=8)

    def drawNimage(self, data, N=20):
        img_data = np.full((28 * N, 28 * N), 255, dtype=np.uint8)
        for pos in range(N * N):
            x_off = (pos // N) * 28
            y_off = (pos  % N) * 28
            for i in range(28):
                for j in range(28):
                    if i == 0 or i == 27 or j == 0 or j == 27:
                        x = x_off + i
                        y = y_off + j
                        img_data[x, y] = 0
                    else:
                        a = data[pos][i * 28 + j]
                        x = x_off + i
                        y = y_off + j
                        value = 255 - a * 255
                        img_data[x, y] = value

        img = Image.fromarray(img_data, mode="L")
        img.show()
        img.save("show.png", quality=95)

    def take_ss(self):
        # remove scaling factor if you are not using scaling (mine is %125, default is %150)
        x1 = self.root.winfo_rootx() + (self.root.winfo_rootx() * 0.25) + 5
        y1 = self.root.winfo_rooty() + (self.root.winfo_rooty() * 0.25) + 5
        x2 = x1 + self.canvas.winfo_width() + (self.canvas.winfo_width() * 0.25) - 5
        y2 = y1 + self.canvas.winfo_height() + (self.canvas.winfo_height() * 0.25) - 5
        self.img_original = ImageGrab.grab().crop((x1, y1, x2, y2))
        self.img = self.img_original.resize((28, 28), Image.ANTIALIAS)

    def button_predict_f(self):
        self.take_ss()

        for i in range(28):
            for j in range(28):
                data = list(self.img.getpixel((j, i)))
                brightness = ceil(0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]) / 255
                self.painted.append(1 - brightness)

        # self.drawNimage(self.painted, N=1)

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

        data = np.array(self.painted).reshape(len(self.painted), 1)
        target = np.array(target).reshape(len(target), 1)
        self.NN.train(data, target, self.test_X.T, self.test_Y.T)

        if self.data_exist:
            self.calculate_accuracy()

        self.combo_box.grid_forget()
        self.label_select_true.grid_forget()
        self.textArea.grid()

        self.button_clear_f()

    def button_clear_f(self):
        self.painted.clear()
        self.img = None
        # self.img_original = None
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
