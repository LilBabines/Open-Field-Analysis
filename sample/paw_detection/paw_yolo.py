import torch
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

def score_frame_track(frame,modelYolo):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """
    
    #frame = [frame]
    results =modelYolo(frame,size=640)
    df=results.pandas().xyxyn[0]
    
    #labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    #return labels, cord
    cofidence=0

    cord=None
    label=None
    for _, row in df.iterrows():

        if row['confidence'] > cofidence :
            cord=[float(row['xmin']), float(row['ymin']),float(row['xmax']), float(row['ymax']),row['confidence']]
            label=row['class']
            cofidence=row['confidence']
            

    return cord,label,cofidence

def plot_boxes_track(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    cord,label = results
    
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    if cord :
        #print(label)
        if label==0:
            #print('n',label)
            c=[0,0,255]
            conf=cord[4]
        else :
            c=[255,0,0]
            #print('e',label)
            conf=-cord[4]

        x1, y1, x2, y2 = int(cord[0]*x_shape), int(cord[1]*y_shape), int(cord[2]*x_shape), int(cord[3]*y_shape)
        #bgr = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        #cv2.putText(frame, str(round(conf,2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
        
    return frame

def load_model(mode='paw'):

    dirYolo=os.path.join(os.getcwd(),'model','paw','yolov5')

    if mode =='paw':

        source_pt=os.path.join(os.getcwd(),'model','poids','best_1920.pt')
    else :
        source_pt=os.path.join(os.getcwd(),'model','poids','LAST_LAST_LAST.pt')

    #print(os.path.exists(source_pt))
    #Yolo=torch.load(yolo_weights_path)
    #yolov5 = torch.hub.load(dirYolo, 'custom', source_pt, source='local')
    model = torch.hub.load(dirYolo, 'custom', path=source_pt, source='local')
    return model

def score_frame(frame,modelYolo):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """
    
    #frame = [frame]
    results =modelYolo(frame,size=1920)
    df=results.pandas().xyxyn[0]
    
    #labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    #return labels, cord
    left_cofidence=0
    right_cofidence=0
    left_cord=None
    left_label=None
    right_cord=None
    right_label=None

    for _, row in df.iterrows():
        #print(row)

        if row['class']==0 or row['class']==2: #left paw
            if row['confidence'] > left_cofidence :
                #print('BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
                left_cord=[float(row['xmin']), float(row['ymin']),float(row['xmax']), float(row['ymax']),row['confidence']]
                left_label=((row['class']//2)-1)*-1
                left_cofidence=row['confidence']

        else : #right paw
            if row['confidence'] > right_cofidence :
                #print('BIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII')
                right_cord=[float(row['xmin']), float(row['ymin']),float(row['xmax']), float(row['ymax']),row['confidence']]
                right_label=(row['class']-1)//2
                right_cofidence=row['confidence']
    #print(left_label,right_label)
    return left_cofidence!=0,right_cofidence!=0,(left_cord,left_label,left_cofidence),(right_cord,right_label,right_cofidence)


def mask(box,img_size=(1080,1920)):
    x_shape, y_shape = img_size[1], img_size[0]
    m=np.zeros(img_size)
    x1, y1, x2, y2 = int(box[0]*x_shape), int(box[1]*y_shape), int(box[2]*x_shape), int(box[3]*y_shape)
    #print(box)
    m[y1:y2+1, x1:x2+1]=1
    #print(type(m))
    #print(m.shape)
    return m

def plot_boxes(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    cord,label = results
    #print(label)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    #if cord :
        
    #print(label)
    if label=='left_yes':
        print("ah")
        #print('n',label)
        c=[0,0,255]
        #conf=cord[4]
    elif label=='right_yes' :
        print("aoh")
        c=[255,0,0]
        #print('e',label)
        ##conf=-cord[4]
    elif label=='left_no':
        print("ooh")
        c=[0,100,100]
    elif label=='right_no':
        print("oooh")
        c=[100,100,0]
    elif label=='elevation' :
        c=[150,20,180]
    elif label=='normal':
        c=[100,100,100]

    x1, y1, x2, y2 = int(cord[0]*x_shape), int(cord[1]*y_shape), int(cord[2]*x_shape), int(cord[3]*y_shape)
    #bgr = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        #cv2.putText(frame, str(round(conf,2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
        
    return frame#,conf

def test(path_to_video,frame_start,cpt,frame_rate_analysis=4):
    Paw_detctoin=load_model(mode='paw')
    Track_model=load_model(mode='pose')


    cap = cv2.VideoCapture(path_to_video)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    i=0
    
    if not cap.isOpened():
        print("Error opening video")

    while(cap.isOpened()):


        status, frame_rgb = cap.read()
        frame=np.copy(frame_rgb)
        
        if status and i%frame_rate_analysis==0:
            #cv2.imshow('ok',frame_rgb)
            cv2.imwrite('image.png',frame_rgb)

            img=Image.open('image.png')
            #print(np.asarray(img))
            result=score_frame_track(img,Track_model)
            plot_boxes(result,frame_rgb)
            if result[1]=='elevation':

            
                retL,retR,scoreL,scoreR=score_frame(img,Paw_detctoin)
                print(retL,retR)
                if retL:
                    #print(score[0])
                    img=plot_boxes(scoreL,frame_rgb)
                #else :
                    #pass
                    #print('no')
                if retR:
                    img=plot_boxes(scoreR,frame_rgb)


            cv2.imshow('yolo',frame_rgb)
            print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
        key = cv2.waitKey(1)
        
        i+=1
        if key == ord('a'):
            
            break
        if key ==ord('f'):
            cv2.imwrite(f'frame{cpt}.png',frame)
            cpt+=1
            
    cap.release()
    cv2.destroyAllWindows()

def run(videos,delays,son,save_dir,yolo_paw,yolo_track,frame_start,frame_rate_analysis):
    
    data_frame=pd.DataFrame(columns=('label_L1','label_L2','label_L3','label_L4','label_R1','label_R2','label_R3','label_R4','cam_left_confidence','cam_right_confidence','num','cam_pose','cam_pose_confidence'))

    cap1 = cv2.VideoCapture(videos[0])
    cap2 = cv2.VideoCapture(videos[1])
    cap3 = cv2.VideoCapture(videos[2])
    cap4 = cv2.VideoCapture(videos[3])


    cap1.set(cv2.CAP_PROP_POS_FRAMES, delays[0]*60+frame_start*son[0]-5)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, delays[1]*60+frame_start*son[1]-1)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, delays[2]*60+frame_start*son[2])
    cap4.set(cv2.CAP_PROP_POS_FRAMES, delays[3]*60+frame_start*son[3])

    max_frame=min([cap1.get(cv2.CAP_PROP_FRAME_COUNT),cap2.get(cv2.CAP_PROP_FRAME_COUNT),cap3.get(cv2.CAP_PROP_FRAME_COUNT),cap4.get(cv2.CAP_PROP_FRAME_COUNT)])
    current_frame=cap1.get(cv2.CAP_PROP_POS_FRAMES)
    bar=tqdm(total=max_frame-current_frame)
    i=0


    while(cap1.isOpened() and current_frame<max_frame):

        current_frame=cap1.get(cv2.CAP_PROP_POS_FRAMES)

        status1, frame1 = cap1.read()
        status2, frame2 = cap2.read()
        status3, frame3 = cap3.read()
        status4, frame4 = cap4.read()
        
        
        if status1 and status2 and status3 and status4 and i%frame_rate_analysis==0:

            frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame3=cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            frame4=cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)

            box1,lab1,conf1=score_frame_track(frame1,yolo_track)
            box2,lab2,conf2=score_frame_track(frame2,yolo_track)
            box3,lab3,conf3=score_frame_track(frame3,yolo_track)
            box4,lab4,conf4=score_frame_track(frame4,yolo_track)
            
            
            retL_1,retR_1,scoreL_1,scoreR_1=None,None,(None,None,None),(None,None,None)
            retL_2,retR_2,scoreL_2,scoreR_2=None,None,(None,None,None),(None,None,None)
            retL_3,retR_3,scoreL_3,scoreR_3=None,None,(None,None,None),(None,None,None)
            retL_4,retR_4,scoreL_4,scoreR_4=None,None,(None,None,None),(None,None,None)
            if bool(lab1) + bool(lab2) +bool(lab3) +bool(lab4)>=2:

            
                retL_1,retR_1,scoreL_1,scoreR_1=score_frame(frame1,yolo_paw)
                retL_2,retR_2,scoreL_2,scoreR_2=score_frame(frame2,yolo_paw)
                retL_3,retR_3,scoreL_3,scoreR_3=score_frame(frame3,yolo_paw)
                retL_4,retR_4,scoreL_4,scoreR_4=score_frame(frame4,yolo_paw)
            
            L1=scoreL_1[1]
            #print(scoreL_1)
            CL1=scoreL_1[2]
           
            
            R1=scoreR_1[1]
            CR1=scoreR_1[2]

        
            L2=scoreL_2[1]
            CL2=scoreL_2[2] 
        
            R2=scoreR_2[1]
            CR2=scoreR_2[2]

        
            L3=scoreL_3[1]
            CL3=scoreL_3[2]  

        
            R3=scoreR_3[1]
            CR3=scoreR_3[2]

        
            L4=scoreL_4[1]
            CL4=scoreL_4[2]
        
            R4=scoreR_4[1]
            CR4=scoreR_4[2]

          #  plot_boxes_track((box1,lab1),frame1)
          #  plot_boxes_track((box2,lab2),frame2)
           # plot_boxes_track((box3,lab3),frame3)
           # plot_boxes_track((box4,lab4),frame4)
           # cv2.imshow('1',frame1)
            #cv2.imshow('2',frame2)
            #cv2.imshow('3',frame3)
            #cv2.imshow('4',frame4)
                
            data_frame.loc[len(data_frame)]={'label_L1':L1,'label_L2':L2,'label_L3':L3,'label_L4':L4,'label_R1':R1,'label_R2':R2,'label_R3':R3,'label_R4':R4,'cam_left_confidence':[CL1,CL2,CL3,CL4],'cam_right_confidence':[CR1,CR2,CR3,CR4],'num':current_frame,'cam_pose':[lab1,lab2,lab3,lab4],'cam_pose_confidence':[conf1,conf2,conf3,conf4]}
            

        
       # key = cv2.waitKey(1)
        
        i+=1
       # if key == ord('a'):
            
        #   break
        bar.update(1)



    data_frame.to_csv(os.path.join(save_dir,'coord.csv'))        
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    bar.close()
    cv2.destroyAllWindows()


def run2(videos,delays,save_dir,yolo_paw,yolo_track,frame_start,frame_rate_analysis):
    
    data_frame=pd.DataFrame(columns=('label_L1','label_L2','label_L3','label_L4','label_R1','label_R2','label_R3','label_R4','cam_left_confidence','cam_right_confidence','num','cam_pose','cam_pose_confidence'))

    cap1 = cv2.VideoCapture(videos[0])
    cap2 = cv2.VideoCapture(videos[1])
    cap3 = cv2.VideoCapture(videos[2])
    cap4 = cv2.VideoCapture(videos[3])


    cap1.set(cv2.CAP_PROP_POS_FRAMES, delays[0]*60+frame_start-5)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, delays[1]*60+frame_start-1)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, delays[2]*60+frame_start)
    cap4.set(cv2.CAP_PROP_POS_FRAMES, delays[3]*60+frame_start)

    max_frame=min([cap1.get(cv2.CAP_PROP_FRAME_COUNT),cap2.get(cv2.CAP_PROP_FRAME_COUNT),cap3.get(cv2.CAP_PROP_FRAME_COUNT),cap4.get(cv2.CAP_PROP_FRAME_COUNT)])
    current_frame=cap1.get(cv2.CAP_PROP_POS_FRAMES)
    bar=tqdm(total=max_frame-current_frame)
    i=0

    cpt=0
    while(cap1.isOpened() and current_frame<max_frame):

        current_frame=cap1.get(cv2.CAP_PROP_POS_FRAMES)

        status1, frame1 = cap1.read()
        status2, frame2 = cap2.read()
        status3, frame3 = cap3.read()
        status4, frame4 = cap4.read()
        f1=np.copy(frame1)
        f2=np.copy(frame2)
        f3=np.copy(frame3)
        f4=np.copy(frame4)
        
        if status1 and status2 and status3 and status4 and i%10==0:

            frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame3=cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            frame4=cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)

            box1,lab1,conf1=score_frame_track(frame1,yolo_track)
            box2,lab2,conf2=score_frame_track(frame2,yolo_track)
            box3,lab3,conf3=score_frame_track(frame3,yolo_track)
            box4,lab4,conf4=score_frame_track(frame4,yolo_track)
            
            
            retL_1,retR_1,scoreL_1,scoreR_1=None,None,(None,None,None),(None,None,None)
            retL_2,retR_2,scoreL_2,scoreR_2=None,None,(None,None,None),(None,None,None)
            retL_3,retR_3,scoreL_3,scoreR_3=None,None,(None,None,None),(None,None,None)
            retL_4,retR_4,scoreL_4,scoreR_4=None,None,(None,None,None),(None,None,None)
            if bool(lab1) + bool(lab2) +bool(lab3) +bool(lab4)>=2:
                cv2.imwrite(os.path.join(save_dir,f'frameA{cpt}.png'),f1)
                cv2.imwrite(os.path.join(save_dir,f'frameB{cpt}.png'),f2)
                cv2.imwrite(os.path.join(save_dir,f'frameC{cpt}.png'),f3)
                cv2.imwrite(os.path.join(save_dir,f'frameD{cpt}.png'),f4)
                cpt+=1
                
        bar.update(1)
        i+=1


    #data_frame.to_csv(os.path.join(save_dir,'coord.csv'))        
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    bar.close()
    cv2.destroyAllWindows()

def test_ele(path_to_video,frame_start,frame_rate_analysis=4):
    #Paw_detctoin=load_model(mode='paw')
    model=load_model(mode='pose')


    cap = cv2.VideoCapture(path_to_video)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    i=0
    
    if not cap.isOpened():
        print("Error opening video")

    while(cap.isOpened()):


        status, frame_rgb = cap.read()
        #frame=np.copy(frame_rgb)
        print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if status and i%frame_rate_analysis==0:
            #cv2.imshow('ok',frame_rgb)
            #cv2.imwrite('image.png',frame_rgb)
            frame1=cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            #img=Image.open('image.png')
            #print(np.asarray(img))
            result=score_frame_track(frame1,model)
            #print(len(result[:2]),'oko')
            plot_boxes_track(result[:2],frame_rgb)
            

            cv2.imshow('yolo',frame_rgb)
            #print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
        key = cv2.waitKey(1)
        
        i+=1
        if key == ord('a'):
            
            break
       
            
    cap.release()
    cv2.destroyAllWindows()

def test_ele_4(delays,sources,frame_start,frame_rate_analysis=4):
    #Paw_detctoin=load_model(mode='paw')
    model=load_model(mode='pose')


    FS=frame_start
    frame_rate_analysis=4
    i=0
    
    cap1,cap2,cap3,cap4=cv2.VideoCapture(sources[0]),cv2.VideoCapture(sources[1]),cv2.VideoCapture(sources[2]),cv2.VideoCapture(sources[3])

    fps=cap1.get(cv2.CAP_PROP_FPS)

    max_frame=min(cap1.get(cv2.CAP_PROP_FRAME_COUNT),cap2.get(cv2.CAP_PROP_FRAME_COUNT),cap3.get(cv2.CAP_PROP_FRAME_COUNT),cap4.get(cv2.CAP_PROP_FRAME_COUNT))

    cap1.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[0]))+ FS)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[1]))+FS )
    cap3.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[2]))+FS )
    cap4.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[3]))+FS )

    

    ret1, frame_rgb1 = cap1.read()
    ret2, frame_rgb2 = cap2.read()
    ret3, frame_rgb3 = cap3.read()
    ret4, frame_rgb4 = cap4.read()
    f1=np.copy(frame_rgb1)
    f2=np.copy(frame_rgb2)
    f3=np.copy(frame_rgb3)
    f4=np.copy(frame_rgb4)
    while(cap1.isOpened() and cap1.get(cv2.CAP_PROP_POS_FRAMES)<max_frame):
        ret1, frame_rgb1 = cap1.read()
        ret2, frame_rgb2 = cap2.read()
        ret3, frame_rgb3 = cap3.read()
        ret4, frame_rgb4 = cap4.read()
        f1=np.copy(frame_rgb1)
        f2=np.copy(frame_rgb2)
        f3=np.copy(frame_rgb3)
        f4=np.copy(frame_rgb4)
        #f=np.copy(frame_rgb1)
        if ret1 and ret2 and ret3 and ret4 and i%frame_rate_analysis==0:
            #cv2.imshow('ok',frame_rgb)
            #cv2.imwrite('image.png',frame_rgb)
            frame1=cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
            frame2=cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
            frame3=cv2.cvtColor(f3, cv2.COLOR_BGR2RGB)
            frame4=cv2.cvtColor(f4, cv2.COLOR_BGR2RGB)
            #img=Image.open('image.png')
            #print(np.asarray(img))
            result1=score_frame_track(frame1,model)
            result2=score_frame_track(frame2,model)
            result3=score_frame_track(frame3,model)
            result4=score_frame_track(frame4,model)
            

            #print(len(result[:2]),'oko')
            plot_boxes_track(result1[:2],frame_rgb1)
            plot_boxes_track(result2[:2],frame_rgb2)
            plot_boxes_track(result3[:2],frame_rgb3)
            plot_boxes_track(result4[:2],frame_rgb4)

            cv2.imshow('yolo1',cv2.resize(frame_rgb1,(1920//2,1080//2),interpolation=cv2.INTER_AREA))
            #cv2.imshow('yolo2',f)
            cv2.imshow('yolo2',cv2.resize(frame_rgb2,(1920//2,1080//2),interpolation=cv2.INTER_AREA))
            cv2.imshow('yolo3',cv2.resize(frame_rgb3,(1920//2,1080//2),interpolation=cv2.INTER_AREA))
            cv2.imshow('yolo4',cv2.resize(frame_rgb4,(1920//2,1080//2),interpolation=cv2.INTER_AREA))


            #print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        i+=1       
        key = cv2.waitKey(1)
        if key==ord('a'):
            break
    cv2.destroyAllWindows()
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()