import tkinter as tk
from tkinter import messagebox
import os

class GUI(object):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Doodle Classifier")
        self.root.geometry("300x350")

        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.names = []
        self.all_names = {}
        self.num_data = 0
        self.data_checkbutton = []

        self.label_number = tk.Label(self.frame, text="number of per data:")
        self.entry_number_area = tk.Entry(self.frame)
        self.label_rate = tk.Label(self.frame, text="learning rate:")
        self.entry_rate_area = tk.Entry(self.frame)
        self.label_epoch = tk.Label(self.frame, text="number of epoch:")
        self.entry_epoch_area = tk.Entry(self.frame)

        self.frame_h = tk.Frame(self.root)

        self.num_layer = [tk.IntVar() for x in range(3)]
        self.num_layer[0].set(1)

        self.c1 = tk.Checkbutton(self.frame_h, text="hidden 1:", onvalue=1, offvalue=1)
        self.c1.select()
        self.hidden_1 = tk.Entry(self.frame_h, width=2)

        self.c2 = tk.Checkbutton(self.frame_h, text="hidden 2:", variable=self.num_layer[1], onvalue=1, offvalue=0,
                                 command=self.layer_selected_control)
        self.hidden_2 = tk.Entry(self.frame_h, width=2)

        self.c3 = tk.Checkbutton(self.frame_h, text="hidden 3:", variable=self.num_layer[2], onvalue=1, offvalue=0,
                                 command=self.show_hidden_3)
        self.hidden_3 = tk.Entry(self.frame_h, width=2)

    def getRoot(self):
        return self.root

    def create_check_buttons(self):
        for file in os.listdir("./data"):
            if file.endswith(".npy"):
                self.all_names[self.num_data] = file[18:-4]
                self.names.append(file[18:-4])
                self.num_data += 1

        self.data_checkbutton = [tk.IntVar() for x in range(self.num_data)]
        for i in range(self.num_data):
            c1 = tk.Checkbutton(self.frame, text=self.all_names[i], variable=self.data_checkbutton[i], onvalue=1,
                                offvalue=0, anchor="w")
            if i % 2 == 0:
                c1.grid(row=i, column=0, sticky="w", ipadx=20)
            else:
                c1.grid(row=i - 1, column=1, sticky="w")

        return self.data_checkbutton, self.all_names, self.num_data

    def layer_selected_control(self):
        if self.num_layer[1].get() == 0:
            self.hidden_2.grid_forget()
            self.hidden_2.delete(0, tk.END)

            self.c3.grid_forget()
            self.hidden_3.grid_forget()
            self.num_layer[2].set(0)
            self.hidden_3.delete(0, tk.END)
        else:
            self.hidden_2.grid(row=0, column=3)
            self.c3.grid(row=0, column=4)

    def show_hidden_3(self):
        if self.num_layer[2].get() == 1:
            self.hidden_3.grid(row=0, column=5)
        else:
            self.hidden_3.grid_forget()
            self.hidden_3.delete(0, tk.END)

    def grid_select_area(self, data_exist, num_data):
        if data_exist:
            self.label_number.grid(row=num_data, column=0, sticky="w")
            self.entry_number_area.grid(row=num_data, column=1, sticky="w", padx=1)
            self.label_rate.grid(row=num_data + 1, column=0, sticky="w")
            self.entry_rate_area.grid(row=num_data + 1, column=1, sticky="w", padx=1)
            self.label_epoch.grid(row=num_data + 2, column=0, sticky="w", padx=1)
            self.entry_epoch_area.grid(row=num_data + 2, column=1, sticky="w", padx=1)

            self.frame_h.pack(fill=tk.BOTH, expand=1)

            self.c1.grid(row=0, column=0)
            self.hidden_1.grid(row=0, column=1)

            self.c2.grid(row=0, column=2)
            # self.hidden_2.grid(row=0, column=3)

            self.c3.grid(row=0, column=4)
            self.hidden_3.grid(row=0, column=5)
            self.c3.grid_remove()
            self.hidden_3.grid_remove()

        else:
            self.label_number.configure(text="Please create file named data and\ncopy data into it")
            self.label_number.grid()

    def get_entries(self):
        valid_entry = True
        marked = False
        message = ""
        num_instance = 1000
        learning_rate = 0.1
        epoch = 1
        hidden_1_size = 64
        hidden_2_size = 32
        hidden_3_size = 16

        for i in range(self.num_data):
            index = self.data_checkbutton[i].get()
            if index == 1:
                marked = True
                self.names.append(self.all_names[i])

        if self.entry_number_area.get() != '':
            if self.is_int(self.entry_number_area.get()):
                num_instance = int(self.entry_number_area.get())
            else:
                valid_entry = False
                message = "write integer"

        if self.entry_rate_area.get() != '':
            if self.is_float(self.entry_rate_area.get()):
                learning_rate = float(self.entry_rate_area.get())
            else:
                valid_entry = False
                message = "write float"

        if self.entry_epoch_area.get() != '':
            if self.is_int(self.entry_epoch_area.get()):
                epoch = int(self.entry_epoch_area.get())
            else:
                valid_entry = False
                message = "write integer"

        if self.hidden_1.get() != '':
            if self.is_int(self.hidden_1.get()):
                hidden_1_size = int(self.hidden_1.get())
            else:
                valid_entry = False
                message = "write integer for layers"

        if self.hidden_2.get() != '':
            if self.is_int(self.hidden_2.get()):
                hidden_2_size = int(self.hidden_2.get())
            else:
                valid_entry = False
                message = "write integer for layers"

        if self.hidden_3.get() != '':
            if self.is_int(self.hidden_3.get()):
                hidden_3_size = int(self.hidden_3.get())
            else:
                valid_entry = False
                message = "write integer for layers"

        all_layers = [hidden_1_size, hidden_2_size, hidden_3_size]
        layers = []
        for i, x in enumerate(all_layers):
            if x != -1 and self.num_layer[i].get() == 1:
                layers.append(x)

        return valid_entry, message, num_instance, learning_rate, epoch, layers

    def data_checkbutton_errors(self, marked, valid_entry, message):
        if not marked and valid_entry:
            self.popup_message("Mark at least 1 image")
            return True
        if not marked and not valid_entry:
            self.popup_message("Mark at least 1 image\n" + message)
            return True
        if marked and not valid_entry:
            self.popup_message(message)
            return True

        return False

    def popup_message(self, message):
        tk.messagebox.showerror("Error", message)

    def is_int(self, value):
        try:
            int(value)
            return True
        except ValueError:
            return False

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def destroy_select_area(self):
        self.frame.destroy()
        self.label_number.destroy()
        self.entry_number_area.destroy()
        self.label_rate.destroy()
        self.entry_rate_area.destroy()
        self.label_epoch.destroy()
        self.entry_epoch_area.destroy()

        self.frame_h.destroy()
        self.c1.destroy()
        self.hidden_1.destroy()
        self.c2.destroy()
        self.hidden_2.destroy()
        self.c3.destroy()
        self.hidden_3.destroy()
