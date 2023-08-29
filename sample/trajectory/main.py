import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
from tqdm import tqdm
import time as t

import argparse

import pandas as pd
import yaml
import sys 
sys.path.insert(1,"sample")
from seg_rat import segmentation_yolo
from syn_video import synchronize
from homography import homography
sys.path.insert(1,"utils")
import read_exel





def test():

    #fichier config (nom des video anomalis, durée, etc.)
    df_config=read_exel.get_df(cfg=DATA_CONFIG)


    #Calcule la trajectoire tout les FRAME_RATE frames
    
    #début d'analyse des video (frame de la camera 1)
    FRAME_START=800

    #vérifie si tout les video du fichier config sont trouvées
    read_exel.verif_video(df_config)

    

    #pour toutes les ligne du fichier config

    row=df_config.sample(1)
    

    #début du chrono
    start = t.time()
    

    
    #on récupère le nom des video (sources), l'expérieince (exp), et le rat (rat)
    sources,exp,rat,title=read_exel.get_info(row,DATA_PATH)

    #on instancie le dataframe de sorti
    data_frame=pd.DataFrame(columns=('x','y','label','num','cam_confidence','mask_score','score1','score2','score3','score4'))

    # on charge homographie
    H1=homography.get(rat,exp,sources[0],DATA_PATH,1)
    H2=homography.get(rat,exp,sources[1],DATA_PATH,2)
    H3=homography.get(rat,exp,sources[2],DATA_PATH,3)
    H4=homography.get(rat,exp,sources[3],DATA_PATH,4)

    #recupère le temps de fin de l'open field (juste avant arrivé de la vitre)
    #time=read_exel.get_time(row)


    #récupère où récuprer les path audio (sur les video MP4 si tout va bien, sinon sur un fichier MP3 dans le même répertoire que la video sans audio [arrive quand video rétournée])
    audios=read_exel.audios(row,DATA_PATH)

    #récupère les anomalie (zoom , lux, son)
    anomalis=read_exel.get_anomalie(row)

    print(rat,exp)


    #chargement du model, YOLO
    yolov5_pose = segmentation_yolo.load_model(MODEL_PATH)

    #on récupère le fps du son
    
    print( '. . . . Synchronization . . . .')
    audio_fps=synchronize.get_fps(sources[3])
    son=[1,1,1,1] #les videos rétournées commencent à la frame 800, pour passer les frame corrompues, donc on avance tout le monde de 800 (framestart) sauf les videos retournées qui sont repérées grace à : ano=='son'
    for i,ano in enumerate(anomalis):
        # si anomalie 'son' le fichier est un MP3 et pas un MP4, il faut donc load MP3 au lieu d'extraire le son de la video, deux fonctoin différentes
        if ano=='son':
            son[i]=0
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
    cap1.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[0]+FRAME_START*son[0]-5)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[1]+FRAME_START*son[1]-1)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[2]+FRAME_START*son[2])
    cap4.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[3]+FRAME_START*son[3])

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
            score1=segmentation_yolo.score_frame(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),yolov5_pose)
            score2=segmentation_yolo.score_frame(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB),yolov5_pose)
            score3=segmentation_yolo.score_frame(cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB),yolov5_pose)
            score4=segmentation_yolo.score_frame(cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB),yolov5_pose)
            
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
    #data_frame.to_csv(os.path.join(SAVE_DIR,'coord.csv'))
    end = t.time()
    print('end',end - start)

def run():
    start = t.time()
    #fichier config (nom des video anomalis, durée, etc.)
    df_config=read_exel.get_df(cfg=DATA_CONFIG)


    #Calcule la trajectoire tout les FRAME_RATE frames
    
    #début d'analyse des video (frame de la camera 1)
    FRAME_START=800

    #vérifie si tout les video du fichier config sont trouvées
    read_exel.verif_video(df_config)

    #chargement du model, YOLO
    yolov5_pose = segmentation_yolo.load_model(MODEL_PATH)

    #pour toutes les ligne du fichier config

    for index,row in df_config.iterrows():

        #début du chrono
        
        

        
        #on récupère le nom des video (sources), l'expérieince (exp), et le rat (rat)
        sources,exp,rat,title=read_exel.get_info(row,DATA_PATH)

        #dossier où l'on range les trajectoires
        current_path=os.path.join(SAVE_PATH,rat,exp)

        if not(os.path.exists(current_path)):
            os.makedirs(current_path)

        #création d'un dossier coordinates{i}, pour i tout les calcul d'une même expérience (ou si deux prélésion d'un même rat par exmeple)
        SAVE_DIR=os.path.join(current_path,'coordinates'+str(len(os.listdir(current_path))))
        os.makedirs(SAVE_DIR)

        #on instancie le dataframe de sorti
        data_frame=pd.DataFrame(columns=('x','y','label','num','cam_confidence','mask_score','score1','score2','score3','score4'))

        # on charge homographie
        H1=homography.get(rat,exp,sources[0],DATA_PATH,1)
        H2=homography.get(rat,exp,sources[1],DATA_PATH,2)
        H3=homography.get(rat,exp,sources[2],DATA_PATH,3)
        H4=homography.get(rat,exp,sources[3],DATA_PATH,4)

        #recupère le temps de fin de l'open field (juste avant arrivé de la vitre)
        #time=read_exel.get_time(row)


        #récupère où récuprer les path audio (sur les video MP4 si tout va bien, sinon sur un fichier MP3 dans le même répertoire que la video sans audio [arrive quand video rétournée])
        audios=read_exel.audios(row,DATA_PATH)

        #récupère les anomalie (zoom , lux, son)
        anomalis=read_exel.get_anomalie(row)

        print(rat,exp)

        #on récupère le fps du son
        
        print( '. . . . Synchronization . . . .')
        audio_fps=synchronize.get_fps(sources[3])


        son=[1,1,1,1] #les videos rétournées commencent à la frame 800, pour passer les frame corrompues, donc on avance tout le monde de 800 (framestart) sauf les videos retournées qui sont repérées grace à : ano=='son'

        for i,ano in enumerate(anomalis):
            # si anomalie 'son' le fichier est un MP3 et pas un MP4, il faut donc load MP3 au lieu d'extraire le son de la video, deux fonctoin différentes
            if ano=='son':
                son[i]=0
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
        cap1.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[0]+FRAME_START*son[0]-5)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[1]+FRAME_START*son[0]-1)
        cap3.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[2]+FRAME_START*son[0])
        cap4.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[3]+FRAME_START*son[0])

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
                score1=segmentation_yolo.score_frame(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),yolov5_pose)
                score2=segmentation_yolo.score_frame(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB),yolov5_pose)
                score3=segmentation_yolo.score_frame(cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB),yolov5_pose)
                score4=segmentation_yolo.score_frame(cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB),yolov5_pose)
                
                k=0
                conf1=conf2=conf3=conf4=0
                mask1=mask2=mask3=mask4=np.zeros((1000,1000))
                if score1[1]!=None:
                    k+=1
                    _,conf1=score1[0][-1]
                    mask1=segmentation_yolo.mask(score1[0])
                    
                if score2[1]!=None:
                    k+=1
                    _,conf2=score2[0][-1]
                    mask2=segmentation_yolo.mask(score2[0])
                    
                if score3[1]!=None:
                    k+=1
                    _,conf3=score3[0][-1]
                    mask3=segmentation_yolo.mask(score3[0])
                    
                if score4[1]!=None:
                    k+=1
                    _,conf4=score4[0][-1]
                    mask4=segmentation_yolo.mask(score4[0])
                    
                
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
                    
                    #cv2.imshow('r',cv2.circle(np.zeros((1000,1000,3)),(round(y_center),round(x_center)),radius=5,thickness=5,color=[255,0,0]))
                    #cv2.imshow('hohoh',homo)

                
                    data_frame.loc[len(data_frame)]={'x':x_center,'y':y_center,'label':lab,'num': current_frame,'cam_confidence':[conf1,conf2,conf3,conf4],'mask_score':mask_score,'score1':score1,'score2':score2,'score3':score3,'score4':score4}
                    #data_frame=pd.concat([data_frame,pd.DataFrame({'x':[x_center],'y':[y_center],'label':lab,'num': current_frame,'cam_confidence':[conf1,conf2,conf3,conf4],'mask_score':mask_score}) ]).reset_index(drop=True)
                    
                #cv2.imshow('mask4',cv2.resize(frame4,(frame4.shape[1]//2,frame4.shape[0]//2)))

                #cv2.imshow('mask1',cv2.resize(frame1,(frame1.shape[1]//2,frame1.shape[0]//2)))
                #cv2.imshow('mask3',cv2.resize(frame3,(frame3.shape[1]//2,frame3.shape[0]//2)))
                #cv2.imshow('mask2',cv2.resize(frame2,(frame2.shape[1]//2,frame2.shape[0]//2)))
            bar.update(1)
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            ret3, frame3 = cap3.read()
            ret4, frame4 = cap4.read()
            ret=ret1 and ret2 and ret3 and ret4
            #key = cv2.waitKey(1)
            
            i+=1
            #if key == ord('a'):

                
                #break
            
        #print(cap1.get(cv2.CAP_PROP_POS_FRAMES))        
        cap1.release()
        cap2.release()
        cap3.release()
        cap4.release()
        cv2.destroyAllWindows()
        bar.close()
        data_frame.to_csv(os.path.join(SAVE_DIR,'coord.csv'))
    end = t.time()
    print('Temps de calcul : ',end - start)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='mode :  "test" pour essayer l algo sur la première expérience avec affichage, "run" pour calculer toute les expérience présente dans le DATA_CONFIG  ',required=True)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    
    with open('./cfg/run_cfg.yaml', 'r') as file :

        dict_cfg = yaml.safe_load(file)

        SAVE_PATH=dict_cfg['SAVE_PATH_TRAJECTORY']
        if not(os.path.exists(SAVE_PATH)):
            print(f"WARNING : save path doesn't not exist , new one created at {SAVE_PATH}")
            os.makedirs(SAVE_PATH)

        DATA_PATH = dict_cfg['DATA_PATH']

        assert os.path.exists(dict_cfg['DATA_PATH']), f" DATA dir doesn't exist, at {DATA_PATH} !! check documentary for set up the projet's hierachy"

        DATA_CONFIG = dict_cfg['DATA_CONFIG']
        assert os.path.exists(dict_cfg['DATA_CONFIG']) ,f" DATA configuration doesn't exist at {DATA_CONFIG} !! check documentary for set up the projet's hierachy"

        FRAME_RATE=dict_cfg['FRAME_RATE']

        MODEL_PATH=dict_cfg['MODEL_PATH']
        assert os.path.exists(dict_cfg['MODEL_PATH']) ,f" MODEL weights path doesn't exist at {MODEL_PATH} !! check documentary for set up the projet's hierachy"

    if opt.mode =='test':
        test()
    elif opt.mode=='run':
        run()
    test()


