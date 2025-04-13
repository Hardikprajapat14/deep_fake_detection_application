# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:27:05 2021

@author: om
"""

import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time
'''import detection_emotion_practice as validate'''

global fn
fn = ""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="Gray")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Fake Image Video Detection System")

# 43

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('gui 10.jpg')
image2 = image2.resize((w,h), Image.Resampling.LANCZOS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  



image2 = Image.open('logo3.png')
image2 = image2.resize((100, 100), Image.LANCZOS)
background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=110,y=160)     


#################################################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def reg():
    from subprocess import call
    call(["python","registration.py"])

def log():
    from subprocess import call
    call(["python","login.py"])
    
def window():
  root.destroy()


button1 = tk.Button(root, text="LOGIN", command=log, width=14, height=1,font=('times', 17, ' bold '), bg="blue", fg="white")
button1.place(x=60, y=300)

button2 = tk.Button(root, text="REGISTER",command=reg,width=14, height=1,font=('times', 17, ' bold '), bg="blue", fg="white")
button2.place(x=60, y=390)


button4 = tk.Button(root, text="Exit",command=window,width=14, height=1,font=('times', 17, ' bold '), bg="Red", fg="white")
button4.place(x=60, y=470)


root.mainloop()
