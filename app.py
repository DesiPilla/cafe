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

from utils import get_face, load_image, resize_with_pad


# create the detector, using default weights
detector = MTCNN()


class SwipingApp():

    def __init__(self, folder):
                
        # Create GUI
        self.gui = tk.Tk(className="Cafe")
        self.gui.minsize(width=100, height=100)
        self.headline = Label(text="Welcome to the Cafe!", font=(None, 25))
        self.headline.pack()

        # Get list of filepaths
        all_files = glob.glob(folder + '/downloaded/*.jpg')
        sip_files = glob.glob(folder + '/sip/*.jpg')
        skip_files = glob.glob(folder + '/skip/*.jpg')
                
        labeled_image_names = [os.path.basename(f) for f in sip_files + skip_files]
        self.image_paths = [f for f in all_files if os.path.basename(f) not in labeled_image_names]
        random.shuffle(self.image_paths)
        
        self.n_sip = len(sip_files)
        self.n_skip = len(skip_files)
        self.remaining = len(self.image_paths)
        
        print("{:,} images already labeled ({:,} sip, {:,} skip), {:,} images still unlabeled.".format(self.n_sip + self.n_skip, self.n_sip, self.n_skip, len(self.image_paths)))
        if not self.image_paths:
            return
        
        # Insert image
        self.sip_or_skip = []          # sip = 1, skip = 0
        
        self.path = self.image_paths[0]
        self.image_paths = self.image_paths[1:]
        self.img = Image.open(self.path).resize((512, 512))
        self.tk_img = ImageTk.PhotoImage(self.img)
        
        self.main_panel = Frame(self.gui)
        self.main_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.panel = Label(self.gui, image=self.tk_img, width=600, height=600)
        self.panel.pack(in_=self.main_panel, side=tk.LEFT)
        
#         self.tally = Frame(self.main_panel)
#         self.tally.pack(in_=self.main_panel, side=tk.RIGHT)
        
        self.sip_count = Label(text = "# Sips: {}\n# Skips: {}\n# Remaining: {}".format(self.n_sip, self.n_skip, self.remaining), font=("Helvetica", 20))
        self.sip_count.pack(in_=self.main_panel, side=tk.RIGHT)
#         self.skip_count = Label(text = "# Skips: {}".format(len(skip_files)), font=("Helvetica", 20))
#         self.sip_count.pack(in_=self.tally, side=tk.TOP)
#         self.skip_count.pack(in_=self.tally, side=tk.BOTTOM)
        
        # Add sip or skip labels
        self.btn_frame = Frame(self.gui)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.btn_skip = Button(self.gui, text="Skip :(", command=self.skip, width=15, height=5, bg='red', fg='white', font=(None, 20))
        self.btn_sip = Button(self.gui, text="Sip :)", command=self.sip, width=15, height=5, bg='green', fg='white', font=(None, 20))
        self.btn_del = Button(self.gui, text="Delete image", command=self.delete, width=15, height=5, bg='black', fg='white', font=(None, 20))
        
        self.btn_skip.pack(in_=self.btn_frame, side=tk.LEFT)
        self.btn_sip.pack(in_=self.btn_frame, side=tk.LEFT)
        self.btn_del.pack(in_=self.btn_frame, side=tk.LEFT)

        # Start GUI
        self.gui.mainloop()
        
    def get_face_and_save(self):
        # Update image
        try:
            face = get_face(np.array(self.img_to_change))[0]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.new_path, face)
        except:
            print("\tERROR: No face was found in", self.path)
            os.remove(self.path)
            os.remove(self.new_path)

        return
        
    def update_image(self, skip=False):
        if self.image_paths:
            self.remaining -= 1
            self.sip_count.configure(text = "# Sips: {}\n# Skips: {}\n# Remaining: {}".format(self.n_sip, self.n_skip, self.remaining))
            if not skip:
                self.get_face_and_save()                            # Crop face of previosuly selected image
                
            self.path = self.image_paths[0]                     # Get path of next image to display
            self.image_paths = self.image_paths[1:]             # Remove from list of remaining paths
            
            self.img = Image.open(self.path).resize((512, 512)) # Open and resize image
            self.tk_img = ImageTk.PhotoImage(self.img)          # Make a Tkinter photo object to display
            self.panel.configure(image=self.tk_img)             # Display image
            self.panel.image = self.tk_img
            
            
        else:
            if not skip:
                self.get_face_and_save()
            
            
            self.headline.destroy()
            self.panel.destroy()
            self.end = Label(text="There are no more images.\nThanks for sipping!", font=(None, 25), height=10)
            self.end.pack()
#             self.btn_frame.destroy()
            self.btn_skip["state"] = "disabled"
            self.btn_sip["state"] = "disabled"
            self.btn_delete["state"] = "disabled"
#             self.gui.destroy()
        
        return
        
    def skip(self):
        print("Skip", self.path)
        self.n_skip += 1
        self.sip_or_skip.append({self.path: 0})
        
        # Copy image to "skip" folder
        fname = os.path.basename(self.path)
        folder = self.path.split(fname)[0]
        self.new_path = folder[:-11] + "/skip/" + fname
        shutil.copy(self.path, self.new_path)
        
        # Copy image to get face later
        self.img_to_change = self.img.copy()
        
        # Change image in GUI
        self.update_image()
        
        # Will crop face after updating GUI
        
    def sip(self):
        print("Sip", self.path)
        self.n_sip += 1
        self.sip_or_skip.append({self.path: 1})
                
        # Copy image to "sip" folder
        fname = os.path.basename(self.path)
        folder = self.path.split(fname)[0]
        self.new_path = folder[:-11] + "/sip/" + fname
        shutil.copy(self.path, self.new_path)
        
        # Copy image to get face later
        self.img_to_change = self.img.copy()
        
        # Change image in GUI
        self.update_image()
        
        # Will crop face after updating GUI
        
    def delete(self):
        print("Delete", self.path)
                
        # Copy image to "sip" folder
        os.remove(self.path)
                
        # Change image in GUI
        self.update_image(skip=True)
        

app = SwipingApp('google_scrape')