# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:18:29 2019

@author: Ravi
"""

import cv2 
import os 

path = "path to the dataset"    
folder_names = os.listdir(path)

for folder in folder_names:
    file_names = os.listdir(path+"/"+folder)

    for file in file_names:
        cam = cv2.VideoCapture(path+"/"+folder+"/"+file) 
        try: 
            # creating a folder named data 
            if not os.path.exists('data'): 
                os.makedirs('data') 
          
        # if not created then raise error 
        except OSError: 
            print ('Error: Creating directory of data') 
          
        # frame 
        currentframe = 0
          
        while(True): 
            f_name = file.split('_')
            # reading from frame 
            print(f_name)
            ret,frame = cam.read() 
            
            if ret: 
                print('name', f_name)
                # if video is still left continue creating images 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                name = './data/'+folder+'/'+str(f_name[2]+"_"+f_name[1])+'-' + str(currentframe) + '.jpg'
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
            


   

