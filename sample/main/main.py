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
from seg_rat import segmentation_yolo
from syn_video import synchronize
from homography import homography
sys.path.insert(1,"utils")
import read_exel


df_config=read_exel.get_df()
DATA_PATH=os.path.join(os.getcwd(),'data')
PATH=os.getcwd()
FRAME_RATE=5
FRAME_START=5000
read_exel.verif_video(df_config)

yolo_weights_path=os.path.join(os.getcwd(),'model','poids','bestPose_Yv7.pt')#,','Utilisateur''auguste.verdier','Desktop','Rat_Tracker','model','yolov5'
yolo_repo=os.path.join(os.getcwd(),'model','yoloEncore','YOLOv7')

yolov7_Pose = segmentation_yolo.load_model(yolo_repo,yolo_weights_path)


for index,row in df_config.iterrows():
    start = t.time()
    print("Début")

    

    sources,exp,rat,title=read_exel.get_info(row,DATA_PATH)

    SAVE_PATH=os.path.join('E:','Stage_Tremplin','TRAJECTORy','resultat',rat,exp)
    SAVE_DIR=os.path.join(SAVE_PATH,'coordinates'+str(len(os.listdir(SAVE_PATH))))
    os.makedirs(SAVE_DIR)
    data_frame=pd.DataFrame(columns=('x','y','label','num','cam_confidence','mask_score','score1','score2','score3','score4'))

    H1=homography.get(os.path.join(DATA_PATH,rat,exp,'homographhy','homo1.npy'))
    H2=homography.get(os.path.join(DATA_PATH,rat,exp,'homographhy','homo2.npy'))
    H3=homography.get(os.path.join(DATA_PATH,rat,exp,'homographhy','homo3.npy'))
    H4=homography.get(os.path.join(DATA_PATH,rat,exp,'homographhy','homo4.npy'))

    time=read_exel.get_time(row)
    print(sources)

    audios=read_exel.audios(row,DATA_PATH)


    anomalis=read_exel.get_anomalie(row)

    print(rat,exp)

     #print(audios)
    audio_fps=synchronize.get_fps(sources[3])
    #print(audio_fps)
    for i,ano in enumerate(anomalis):
        #print(i)
        if ano=='son':
            #print(audios[i])
            #print(os.path.exists(audios[i]))
            audios[i]=synchronize.load_audio(audios[i])
            pass
        else :
            #print(os.path.exists(audios[i]))
            audios[i]=synchronize.get_audio(audios[i])
            pass

    #delays=[1494.0/60, 1403.0/60, 1220.0/60, 1110.0/60]
    delays=synchronize.global_delay2(audios,fps=audio_fps)
    #synchronize.delay_run_4(sources,delays,framestart=500)
    #print(delays)

    cap1=cv2.VideoCapture(sources[0])
    cap2=cv2.VideoCapture(sources[1])
    cap3=cv2.VideoCapture(sources[2])
    cap4=cv2.VideoCapture(sources[3])

    fps=cap1.get(cv2.CAP_PROP_FPS)
        
    cap1.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[0]+FRAME_START)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[1]+FRAME_START)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[2]+FRAME_START)
    cap4.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[3]+FRAME_START)

    max_frame=min([cap1.get(cv2.CAP_PROP_FRAME_COUNT),cap2.get(cv2.CAP_PROP_FRAME_COUNT),cap3.get(cv2.CAP_PROP_FRAME_COUNT),cap4.get(cv2.CAP_PROP_FRAME_COUNT)])
    print('end at ',max_frame)
    
    
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
        #print('a')
        if ret and i%FRAME_RATE==0:
            #frame1=np.copy(frame_rgb_1)
            #frame2=np.copy(frame_rgb_2)
            #frame3=np.copy(frame_rgb_3)
            #frame4=np.copy(frame_rgb_4)
            #cv2.imshow('ok',frame1)

            #cv2.imwrite('image1.png',frame1)
            #img1=Image.open('image1.png')

            #cv2.imwrite('image2.png',frame2)
            #img2=Image.open('image2.png')

            #cv2.imwrite('image3.png',frame3)
            #img3=Image.open('image3.png')

            #cv2.imwrite('image4.png',frame4)
            #img4=Image.open('image4.png')
            #print(np.asarray(img))
            
            score1=segmentation_yolo.score_frame(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),yolov7_Pose)
            score2=segmentation_yolo.score_frame(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB),yolov7_Pose)
            score3=segmentation_yolo.score_frame(cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB),yolov7_Pose)
            score4=segmentation_yolo.score_frame(cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB),yolov7_Pose)
            #print(frame_rgb.shape)
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
    




