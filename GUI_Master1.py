import tkinter as tk
from PIL import Image , ImageTk
import csv
from datetime import date
import time
import numpy as np
import cv2
from tkinter.filedialog import askopenfilename
import os
from tkinter import messagebox as ms
import shutil
#from skimage import measure
#import Train_FDD_cnn as TrainM
global fn

#==============================================================================
root = tk.Tk()
root.state('zoomed')

root.title("Fake Image Video Detection System")

current_path = str(os.path.dirname(os.path.realpath('__file__')))

basepath=current_path  + "\\" 

#==============================================================================
#==============================================================================
#Background Image Setup
img = Image.open(basepath + "gui 11.jpg")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()

bg = img.resize((w,h),Image.LANCZOS)

bg_img = ImageTk.PhotoImage(bg)

bg_lbl = tk.Label(root,image=bg_img)
bg_lbl.place(x=0,y=0)


     
def show_FDD_video(video_path):
    ''' Display FDD video with annotated bounding box and labels '''
    from keras.models import load_model
    
    
    img_cols, img_rows = 64,64
    
    FALLModel=load_model(r'fake_event.h5')   
    
    video = cv2.VideoCapture(video_path)
        
    

    if (not video.isOpened()):
        print("{} cannot be opened".format(video_path))
        # return False

    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    red = (0, 0, 255)
    
    line_type = cv2.LINE_AA
    i=1
    
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        img=cv2.resize(frame,(img_cols, img_rows),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
        
        X_img = img.reshape(-1, img_cols, img_rows, 1)
        X_img = X_img.astype('float32')
        
        X_img /= 255
        
        predicted =FALLModel.predict(X_img)

        if predicted[0][0] < 0.5:
            predicted[0][0] = 0
            predicted[0][1] = 1
            label = 1
        else:
            predicted[0][0] = 1
            predicted[0][1] = 0
            label = 0
          
        frame_num = int(i) 
        
        label_text = ""
        
        color = (255, 255, 255)
        
        if  label == 1 :
            label_text = "Fake Image Detected"
            color = red
        else:
            label_text = "Normal Image Detected"
            color = green

        frame = cv2.putText(
            frame, "Frame: {}".format(frame_num), (5, 30),
            fontFace = font, fontScale = 1, color = color, lineType = line_type
        )
        frame = cv2.putText(
            frame, "Label: {}".format(label_text), (5, 60),
            fontFace = font, fontScale =1, color = color, lineType = line_type
        )

        i=i+1
        cv2.imshow('FDD', frame)
        if cv2.waitKey(30) == 27:
            break

    video.release()
    cv2.destroyAllWindows()
       
    
def Video_Verify():
    
    global fn
    

    fileName = askopenfilename(initialdir='/dataset', title='Select image',
                               filetypes=[("all files", "*.*")])

    fn = fileName
    Sel_F = fileName.split('/').pop()
    Sel_F = Sel_F.split('.').pop(1)
            
    if Sel_F!= 'mp4':
        print("Select Video File!!!!!!")
    else:
        
       
        show_FDD_video(fn)
       
#============================================================================================================
#upload video
def upload():
     
    global fn

    fileName = askopenfilename(initialdir='/dataset', title='Select image',
                               filetypes=[("all files", "*.*")])

    fn = fileName
    Sel_F = fileName.split('/').pop()
    Sel_F = Sel_F.split('.').pop(1)
            
    if Sel_F!= 'mp4':
        print("Select Video File!!!!!!")
        ms.showerror('Oops!', 'Select Video File!!!!!!')
    else:
        ms.showinfo('Success!', 'Video Uploaded Successfully !')
        return fn

#upload image
def upload1():
     
    global fn
    from keras.models import load_model

    fileName = askopenfilename(initialdir='/dataset', title='Select image', filetypes=[("all files", "*.*")])

    
    
    img_cols, img_rows = 64,64
    
    FALLModel=load_model(r'fake_event.h5')    
    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    red = (0, 0, 255)
   
    line_type = cv2.LINE_AA
    i=1

    frame = cv2.imread(fileName)
    img=cv2.resize(frame,(img_cols, img_rows),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    
    X_img = img.reshape(-1, img_cols, img_rows, 1)
    X_img = X_img.astype('float32')
    
    X_img /= 255
    
    predicted =FALLModel.predict(X_img)

    if predicted[0][0] < 0.5:
        predicted[0][0] = 0
        predicted[0][1] = 1
        label = 1
    else:
        predicted[0][0] = 1
        predicted[0][1] = 0
        label = 0
      
    frame_num = int(i)  
  
    label_text = ""
    
    color = (255, 255, 255)
    
    if  label == 1 :
        label_text = "Fake Image Detected"
        color = red
    else:
        label_text = "Normal Image Detected"
        color = green

    frame = cv2.putText(
        frame, "Frame: {}".format(frame_num), (5, 30),
        fontFace = font, fontScale = 1, color = color, lineType = line_type
    )
    frame = cv2.putText(
        frame, "Label: {}".format(label_text), (5, 60),
        fontFace = font, fontScale =1, color = color, lineType = line_type
    )

    cv2.imshow('FDD', frame)
    

#convert video into frames    
def convert():
#    global fn
    #a=upload(fn)
    cam = cv2.VideoCapture(fn)
    try:
          
        # creating a folder named data
        if not os.path.exists('images'):
            os.makedirs('images')
      
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of images')
      
    # frame
    currentframe = 0
      
    while(True):
          
        # reading from frame
        ret,frame = cam.read()
      
        if ret:
            # if video is still left continue creating images
            name = './images/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
      
            # writing the extracted images
            cv2.imwrite(name, frame)
      
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
      
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    ms.showinfo('Success!', 'Video converted into frames Successfully !')

def CLOSE():
    root.destroy()
    
   
###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            

button5 = tk.Button(root,command = upload, text="Upload Video", width=22,font=("Times new roman", 17, "bold"),bg="black",fg="white")
button5.place(x=10,y=20)

button5 = tk.Button(root,command = upload1, text="Upload Image", width=22,font=("Times new roman", 17, "bold"),bg="black",fg="white")
button5.place(x=10,y=100)

button1 = tk.Button(root,command = convert, text="Convert Video To Frames", width=22,font=("Times new roman", 17, "bold"),bg="black",fg="white")
button1.place(x=400,y=20)

button2 = tk.Button(root,command = Video_Verify, text="Detect Fake Video", width=22,font=("Times new roman", 17, "bold"),bg="black",fg="white")
button2.place(x=800,y=20)


close = tk.Button(root,command = CLOSE, text="Exit", width=22,font=("Times new roman", 17, "bold"),bg="red",fg="white")
close.place(x=1200,y=20)


root.mainloop()






