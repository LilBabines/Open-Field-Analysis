import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
from tqdm import tqdm
import time as t

import pandas as pd

import sys 
sys.path.insert(1,"sample")
from seg_rat import segmentation_yolo
from syn_video import synchronize
from homography import homography
sys.path.insert(1,"utils")
import read_exel

#fichier config (nom des video anomalis, durée, etc.)
df_config=read_exel.get_df()

#emplacement des donées
DATA_PATH=os.path.join(os.getcwd(),'data')

#dossier courant
PATH=os.getcwd()

#Calcule la trajectoire tout les FRAME_RATE frames
FRAME_RATE=5

#début d'analyse des video (frame de la camera 1)
FRAME_START=5000

#vérifie si tout les video du fichier config sont trouvées
read_exel.verif_video(df_config)

#chargement du model, YOLO
yolo_weights_path=os.path.join(os.getcwd(),'model','poids','bestPose_Yv7.pt')#,','Utilisateur''auguste.verdier','Desktop','Rat_Tracker','model','yolov5'
yolo_repo=os.path.join(os.getcwd(),'model','yoloEncore','YOLOv7')

yolov7_Pose = segmentation_yolo.load_model(yolo_repo,yolo_weights_path)

#pour toutes les ligne du fichier config
for index,row in df_config.iterrows():

    #début du chrono
    start = t.time()
    print("Début")

    
    #on récupère le nom des video (sources), l'expérieince (exp), et le rat (rat)
    sources,exp,rat,title=read_exel.get_info(row,DATA_PATH)

    #dossier où l'on range les trajectoires
    SAVE_PATH=os.path.join('E:','Stage_Tremplin','TRAJECTORy','resultat',rat,exp)

    #création d'un dossier coordinates{i}, pour i tout les calcul d'une même expérience (ou si deux prélésion d'un même rat par exmeple)
    SAVE_DIR=os.path.join(SAVE_PATH,'coordinates'+str(len(os.listdir(SAVE_PATH))))
    os.makedirs(SAVE_DIR)

    #on instancie le dataframe de sorti
    data_frame=pd.DataFrame(columns=('x','y','label','num','cam_confidence','mask_score','score1','score2','score3','score4'))

    # on charge homographie
    H1=homography.get(rat,exp,sources[0],DATA_PATH,0)
    H2=homography.get(rat,exp,sources[1],DATA_PATH,1)
    H3=homography.get(rat,exp,sources[2],DATA_PATH,2)
    H4=homography.get(rat,exp,sources[3],DATA_PATH,3)

    #recupère le temps de fin de l'open field (juste avant arrivé de la vitre)
    #time=read_exel.get_time(row)


    #récupère où récuprer les path audio (sur les video MP4 si tout va bien, sinon sur un fichier MP3 dans le même répertoire que la video sans audio [arrive quand video rétournée])
    audios=read_exel.audios(row,DATA_PATH)

    #récupère les anomalie (zoom , lux, son)
    anomalis=read_exel.get_anomalie(row)

    print(rat,exp)

    #on récupère le fps du son
    audio_fps=synchronize.get_fps(sources[3])

    for i,ano in enumerate(anomalis):
        # si anomalie 'son' le fichier est un MP3 et pas un MP4, il faut donc load MP3 au lieu d'extraire le son de la video, deux fonctoin différentes
        if ano=='son':
            
            audios[i]=synchronize.load_audio(audios[i])
            
        else :
            
            audios[i]=synchronize.get_audio(audios[i])
            

    #calcul du délays entre video, en seconde
    delays=synchronize.global_delay2(audios,fps=audio_fps)


    cap1=cv2.VideoCapture(sources[0])
    cap2=cv2.VideoCapture(sources[1])
    cap3=cv2.VideoCapture(sources[2])
    cap4=cv2.VideoCapture(sources[3])

    fps=cap1.get(cv2.CAP_PROP_FPS)

    #on fait commencer les videos à frame start + délai   
    cap1.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[0]+FRAME_START)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[1]+FRAME_START)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[2]+FRAME_START)
    cap4.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[3]+FRAME_START)

    # calcul la fin d'expérience
    max_frame=min([cap1.get(cv2.CAP_PROP_FRAME_COUNT),cap2.get(cv2.CAP_PROP_FRAME_COUNT),cap3.get(cv2.CAP_PROP_FRAME_COUNT),cap4.get(cv2.CAP_PROP_FRAME_COUNT)])
    #print('end at ',max_frame)
    
    
    ret1,frame1=cap1.read()
    ret2,frame2=cap2.read()
    ret3,frame3=cap3.read()
    ret4,frame4=cap4.read()
    ret= ret1 and ret2 and ret3 and ret4
    current_frame=cap1.get(cv2.CAP_PROP_POS_FRAMES)
    bar=tqdm(total=max_frame-current_frame)
    
    i=0
    while(cap1.isOpened() and current_frame<max_frame) :

        current_frame=cap1.get(cv2.CAP_PROP_POS_FRAMES)
       
        if ret and i%FRAME_RATE==0:
           
            #calcul prédictions
            score1=segmentation_yolo.score_frame(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),yolov7_Pose)
            score2=segmentation_yolo.score_frame(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB),yolov7_Pose)
            score3=segmentation_yolo.score_frame(cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB),yolov7_Pose)
            score4=segmentation_yolo.score_frame(cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB),yolov7_Pose)
            
            k=0
            conf1=conf2=conf3=conf4=0
            mask1=mask2=mask3=mask4=np.zeros((1000,1000))
            if score1[1]!=None:
                #print(score[0])
                k+=1
                _,conf1=segmentation_yolo.plot_boxes(score1,frame1)
                mask1=segmentation_yolo.mask(score1[0])
                #x1,y1,x2,y2=score[1][0:4]
            if score2[1]!=None:
                #print(score[0])
                k+=1
                _,conf2=segmentation_yolo.plot_boxes(score2,frame2)
                mask2=segmentation_yolo.mask(score2[0])
                #x1,y1,x2,y2=score[1][0:4]
            if score3[1]!=None:
                k+=1
                #print(score[0])
                _,conf3=segmentation_yolo.plot_boxes(score3,frame3)
                mask3=segmentation_yolo.mask(score3[0])
                #x1,y1,x2,y2=score[1][0:4]
            if score4[1]!=None:
                k+=1
                #print(score[0])
                _,conf4=segmentation_yolo.plot_boxes(score4,frame4)
                mask4=segmentation_yolo.mask(score4[0])
                #x1,y1,x2,y2=score[1][0:4]
            #print(conf1,conf2,conf3,conf4)
            if k>=2:

                
                conf=conf1+conf2+conf3+conf4
                homo1=cv2.warpPerspective(mask1, H1,(1000, 1000))
                homo2=cv2.warpPerspective(mask2, H2,(1000, 1000))
                homo3=cv2.warpPerspective(mask3, H3,(1000, 1000))
                homo4=cv2.warpPerspective(mask4, H4,(1000, 1000))
                homo=(homo1 + homo2+ homo3+ homo4)/k
                mask_score=np.max(homo)
                l=np.argwhere(homo==mask_score)
                x_center, y_center = l.sum(0)/len(l)
                #print(x_center,y_center)
                #print(np.max(homo))
                #cv2.imwrite('homo.png',homo*255)
                if conf>0:
                    lab='normal'
                else :
                    lab='elevation'
                #cv2.imshow('yolo',cv2.putText(frame1,lab,(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,[0,255,0],2,cv2.LINE_AA))
                
                cv2.imshow('r',cv2.circle(np.zeros((1000,1000,3)),(round(y_center),round(x_center)),radius=5,thickness=5,color=[255,0,0]))
                cv2.imshow('hohoh',homo)

            #,'label','num','cam_confidence','mask_score'
                data_frame.loc[len(data_frame)]={'x':x_center,'y':y_center,'label':lab,'num': current_frame,'cam_confidence':[conf1,conf2,conf3,conf4],'mask_score':mask_score,'score1':score1,'score2':score2,'score3':score3,'score4':score4}
                #data_frame=pd.concat([data_frame,pd.DataFrame({'x':[x_center],'y':[y_center],'label':lab,'num': current_frame,'cam_confidence':[conf1,conf2,conf3,conf4],'mask_score':mask_score}) ]).reset_index(drop=True)
                
            cv2.imshow('mask4',cv2.resize(frame4,(frame4.shape[1]//2,frame4.shape[0]//2)))

            cv2.imshow('mask1',cv2.resize(frame1,(frame1.shape[1]//2,frame1.shape[0]//2)))
            cv2.imshow('mask3',cv2.resize(frame3,(frame3.shape[1]//2,frame3.shape[0]//2)))
            cv2.imshow('mask2',cv2.resize(frame2,(frame2.shape[1]//2,frame2.shape[0]//2)))
        bar.update(1)
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()
        ret=ret1 and ret2 and ret3 and ret4
        key = cv2.waitKey(1)
        
        i+=1
        if key == ord('a'):

            
            break
         
    print(cap1.get(cv2.CAP_PROP_POS_FRAMES))        
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    cv2.destroyAllWindows()
    bar.close()
    data_frame.to_csv(os.path.join(SAVE_DIR,'coord.csv'))
    end = t.time()
    print('end',end - start)
    




