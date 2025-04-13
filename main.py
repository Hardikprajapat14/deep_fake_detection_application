# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:24:42 2024

@author: COMPUY
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
#import video_capture as value
#import lecture_details as detail_data
#import video_second as video1

#import lecture_video  as video

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
image2 = Image.open('Deepfake main.png')
image2 = image2.resize((w,h), Image.LANCZOS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  



#################################################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




def log():
    from subprocess import call
    call(["python","GUI_Master1.py"])
    
def window():
  root.destroy()


button1 = tk.Button(root, text="Deep Fake Detection", command=log, width=20, height=1,font=('times', 17, ' bold '), bg="yellow", fg="black")
button1.place(x=50, y=100)




button4 = tk.Button(root, text="Exit",command=window,width=14, height=1,font=('times', 17, ' bold '), bg="Red", fg="white")
button4.place(x=80, y=200)


root.mainloop()
