import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
from tqdm import tqdm
import time as t


from IPython.display import display
import pandas as pd
from PIL import Image
import sys 
sys.path.insert(1,"sample")
from paw_detection import paw_yolo
from syn_video import synchronize
sys.path.insert(1,"utils")
import read_exel


df_config=read_exel.get_df()
DATA_PATH=os.path.join(os.getcwd(),'data')
PATH=os.getcwd()
FRAME_RATE=1
FRAME_START=800
read_exel.verif_video(df_config)

Paw_detction=paw_yolo.load_model(mode='paw')
Track_model=paw_yolo.load_model(mode='pose')

for index,row in df_config.iterrows():
    start = t.time()
    print("Début")

    

    sources,exp,rat,title=read_exel.get_info(row,DATA_PATH)

    SAVE_PATH=os.path.join('E:','Stage_Tremplin','PAW','resultat2',rat,exp)
    if not(os.path.exists(SAVE_PATH)):
        os.makedirs(SAVE_PATH)
    
    SAVE_DIR=os.path.join(SAVE_PATH,'coordinates'+str(len(os.listdir(SAVE_PATH))))
    os.makedirs(SAVE_DIR)


 
    time=read_exel.get_time(row)


    audios=read_exel.audios(row,DATA_PATH)


    anomalis=read_exel.get_anomalie(row)
    son=[1,1,1,1]
    print(rat,exp)


    audio_fps=synchronize.get_fps(sources[3])
    #print(audio_fps)
    for i,ano in enumerate(anomalis):
        #print(i)
        if ano=='son':
            #print(audios[i])
            #print(os.path.exists(audios[i]))
            audios[i]=synchronize.load_audio(audios[i])
            son[i]=0
            #pass
        else :
            #print(os.path.exists(audios[i]))
            audios[i]=synchronize.get_audio(audios[i])
            #pass

    #delays=[1494.0/60, 1403.0/60, 1220.0/60, 1110.0/60]
    delays=synchronize.global_delay2(audios,fps=audio_fps)
    
    
    
    paw_yolo.run(sources,delays,son,SAVE_DIR,Paw_detction,Track_model,frame_start=FRAME_START,frame_rate_analysis=FRAME_RATE)

    
    end = t.time()
    print('end',end - start)
    




