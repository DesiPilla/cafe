import os
import cv2
import time

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageTk

import tkinter as tk
from tkinter import Label, Button, PhotoImage, Frame


files = ['train_images/{}'.format(f) for f in os.listdir('./train_images')]
friends = ['friends/{}'.format(f) for f in os.listdir('./friends')]
random.shuffle(friends)


class SwipingApp():

    def __init__(self, files):
        
        # Create GUI
        self.gui = tk.Tk(className="Cafe")
        self.gui.minsize(width=100, height=100)
        self.headline = Label(text="Welcome to the Cafe!", font=(None, 25))
        self.headline.pack()

        # Insert image
        self.image_paths = files
        self.sip_or_skip = []          # sip = 1, skip = 0
        
        path = self.image_paths[0]
        self.image_paths = self.image_paths[1:]
        im = Image.open(path).resize((512, 512))
        im = ImageTk.PhotoImage(im)
        self.panel = Label(image=im)
        self.panel.pack()
        
                
        # Add sip or skip labels
        self.btn_frame = Frame(self.gui)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.btn_skip = Button(self.gui, text="Skip :(", command=self.skip, width=25, height=5, bg='red', fg='white', font=(None, 20))
        self.btn_sip = Button(self.gui, text="Sip :)", command=self.sip, width=25, height=5, bg='green', fg='white', font=(None, 20))
        
        self.btn_skip.pack(in_=self.btn_frame, side=tk.LEFT)
        self.btn_sip.pack(in_=self.btn_frame, side=tk.LEFT)

        # Start GUI
        self.gui.mainloop()
        
    def update_image(self):
#         print(self.image_paths)
        if self.image_paths:
            path = self.image_paths[0]
            self.image_paths = self.image_paths[1:]
            im = Image.open(path).resize((512, 512))
            im = ImageTk.PhotoImage(im)
            self.panel.configure(image=im)
            self.panel.image = im
        else:
            self.headline.destroy()
            self.panel.destroy()
            self.end = Label(text="There are no more images.\nThanks for sipping!", font=(None, 25), height=10)
            self.end.pack()
#             self.btn_frame.destroy()
            self.btn_skip["state"] = "disabled"
            self.btn_sip["state"] = "disabled"
#             self.gui.destroy()
        
        return
        
    def skip(self):
        print("Skip")
        self.sip_or_skip.append(0)
        self.update_image()
        
    def sip(self):
        print("Sip")
        self.sip_or_skip.append(1)
        self.update_image()
        
        
        
app = SwipingApp(files[:5] + friends)