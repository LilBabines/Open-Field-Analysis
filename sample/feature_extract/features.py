import os 
import cv2
import numpy as np

from tqdm import tqdm
import tensorflow as tf
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import sys 
sys.path.insert(1,"sample")
from seg_rat import segmentation
from syn_video import synchronize
from homography import homography

class Rat_tracker():

    def __init__(self,video_sources,exp,rat,title,audio1=None,audio2=None,audio3=None,audio4=None,source_model=os.path.join("model","model3.h5"),homographies=None,dir_coord_save=os.path.join( "data","EXP 1","coordinates")):
        

        self.videos=video_sources
        self.video1=self.videos[0]
        self.video2=self.videos[1]
        self.video3=self.videos[2]
        self.video4=self.videos[3]
        save_path=os.path.join('E:','Stage Tremplin','TRAJECTORy','resultat',rat,exp)

        self.dir_coord_save=os.path.join(save_path,'coordinates'+str(len(os.listdir(save_path))))
        os.makedirs(os.path.join(save_path,'coordinates'+str(len(os.listdir(save_path)))))
        self.title_plot=title
        print(". . . . Load Model . . . .")
        self.model=segmentation.load_model(source_model)
        print(". . . . Load Audio . . . .")
        self.audio_fps=synchronize.get_fps(self.videos[3])
        if audio1 != None :
            print(". . . Audio 1 . . .")
            audio1=synchronize.load_audio(audio1)
        else :
            print(". . . Audio 1 . . .")
            audio1= synchronize.get_audio(self.videos[0])

        if audio2 != None :
            print(". . . Audio 2 . . .")
            audio2=synchronize.load_audio(audio2)
        else :
            print(". . . Audio 2 . . .")
            audio2= synchronize.get_audio(self.videos[1])
        
        if audio3 != None :    
            print(". . . Audio 3 . . .")
            audio3=synchronize.load_audio(audio3)
        else :
            print(". . . Audio 3 . . .")
            audio3= synchronize.get_audio(self.videos[2])

        if audio4 != None :
            print(". . . Audio 4 . . .")
            audio4=synchronize.load_audio(audio4)
        else :
            print(". . . Audio 4 . . .")
            audio4= synchronize.get_audio(self.videos[3])


        audios=[audio1,audio2,audio3,audio4]
        print(" . .  Synchronize Audio . .")
        self.video_delay=synchronize.global_delay(audios,self.audio_fps)
        print(". .  Build Homographies . .")
        homo_path=os.path.join(os.getcwd(),rat,exp,'homography')
        if os.path.exists(os.path.join(homo_path,"homo1.npy")):
            self.H1=np.load(os.path.exists(os.path.join(homo_path,"homo1.npy")))
        else :
            self.H1=homography.findHomography(self.video1)

        if os.path.exists(os.path.join(homo_path,"homo2.npy")):
            self.H1=np.load(os.path.exists(os.path.join(homo_path,"homo2.npy")))
        else :
            self.H1=homography.findHomography(self.video2)

        if os.path.exists(os.path.join(homo_path,"homo3.npy")):
            self.H1=np.load(os.path.exists(os.path.join(homo_path,"homo3.npy")))
        else :
            self.H1=homography.findHomography(self.video3)
        if os.path.exists(os.path.join(homo_path,"homo4.npy")):
            self.H1=np.load(os.path.exists(os.path.join(homo_path,"homo4.npy")))
        else :
            self.H1=homography.findHomography(self.video4)


        print("----------------OKAY---------------")


    def load(self,sec_analyse,frame_start=800,frame_rate_analysis=5,size_tronc=300):
        
        self.coordinate=pd.DataFrame(columns=('x','y','num'))

        cap1 = cv2.VideoCapture(self.video1)
        cap2 = cv2.VideoCapture(self.video2)
        cap3 = cv2.VideoCapture(self.video3)
        cap4 = cv2.VideoCapture(self.video4)

        fps=cap1.get(cv2.CAP_PROP_FPS)

        total_frame=min(cap1.get(cv2.CAP_PROP_FRAME_COUNT),cap2.get(cv2.CAP_PROP_FRAME_COUNT),cap3.get(cv2.CAP_PROP_FRAME_COUNT),cap4.get(cv2.CAP_PROP_FRAME_COUNT))

        
        print("total frame",total_frame)

        number_frame_analyse=min((sec_analyse*fps)/frame_rate_analysis,(total_frame-(fps*self.video_delay[3]+frame_start))/frame_rate_analysis)
        number_frame_pass=min((sec_analyse*fps),(total_frame-(fps*self.video_delay[3]+frame_start)))
        print(number_frame_pass)
        print("frame analysées",number_frame_analyse)

        nombre_tronc=number_frame_analyse//size_tronc +1
        
        print("nombre troncon necessaire :", nombre_tronc)
        for i in range( int(nombre_tronc)-1):
            #print(int(frame_start+i*size_tronc*frame_rate_analysis),int(frame_start+(i+1)*size_tronc*frame_rate_analysis))
            
            self.tronc(int(frame_start+i*size_tronc*frame_rate_analysis),int(frame_start+(i+1)*size_tronc*frame_rate_analysis),frame_rate_analysis)
            self.extract_trajectory()
        
        #print(int(frame_start+(nombre_tronc-1)*size_tronc*frame_rate_analysis),number_frame_pass+frame_start)
        self.tronc(int(frame_start+(nombre_tronc-1)*size_tronc*frame_rate_analysis),number_frame_pass+frame_start,frame_rate_analysis)
        self.extract_trajectory()
        self.plot()
        self.coordinate.to_csv(os.path.join(self.dir_coord_save,'coord.csv'))


    def tronc(self,start_frame,end_frame,frame_rate_analysis):
        number_frame_analyse=int((end_frame-start_frame)//frame_rate_analysis)
 
        
        print(". . . . SetUp / Allocation . . . .")
        cap1 = cv2.VideoCapture(self.video1)
        cap2 = cv2.VideoCapture(self.video2)
        cap3 = cv2.VideoCapture(self.video3)
        cap4 = cv2.VideoCapture(self.video4)
        fps=cap1.get(cv2.CAP_PROP_FPS)
        
        cap1.set(cv2.CAP_PROP_POS_FRAMES, fps*self.video_delay[0]+start_frame)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, fps*self.video_delay[1]+start_frame)
        cap3.set(cv2.CAP_PROP_POS_FRAMES, fps*self.video_delay[2]+start_frame)
        cap4.set(cv2.CAP_PROP_POS_FRAMES, fps*self.video_delay[3]+start_frame)
        print("départ",cap1.get(cv2.CAP_PROP_POS_FRAMES))
        i=0
        #width  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
        #print( (number_frame_analyse,height,width ,3))
        #self.frame=np.zeros((number_frame_analyse,height,width ,3),dtype=np.uint8)
    
        frames1=np.zeros((number_frame_analyse,320, 448))
 
        frames2=np.zeros((number_frame_analyse,320, 448))



        frames3=np.zeros((number_frame_analyse,320, 448))


        frames4=np.zeros((number_frame_analyse,320, 448))

        self.num_frame=np.zeros(number_frame_analyse) -1


        print(". . . . Load Frames . . . . ")
        
        frame_count=0
        
        if not cap1.isOpened():
            print("Error opening video")


        bar=tqdm(total=number_frame_analyse)

        
        while i<number_frame_analyse:
            status1, frame_rgb_1 = cap1.read()
            status2, frame_rgb_2 = cap2.read()
            status3, frame_rgb_3 = cap3.read()
            status4, frame_rgb_4 = cap4.read()
            
            

            
            if frame_count%frame_rate_analysis==0 :
                if status1 and  status2 and status3 and status4:
                
                
                
                    frame1 = cv2.cvtColor(frame_rgb_1, cv2.COLOR_BGR2GRAY)
                    frame2 = cv2.cvtColor(frame_rgb_2, cv2.COLOR_BGR2GRAY)
                    frame3 = cv2.cvtColor(frame_rgb_3, cv2.COLOR_BGR2GRAY)
                    frame4 = cv2.cvtColor(frame_rgb_4, cv2.COLOR_BGR2GRAY)
                    
                    # x1=np.array(cv2.resize(frame1, (448, 320), interpolation = cv2.INTER_AREA))
                    # #x1=np.expand_dims(x1,axis=(2))
                    # x2=np.array(cv2.resize(frame2, (448, 320), interpolation = cv2.INTER_AREA))
                    # #x2=np.expand_dims(x2,axis=(2))
                    # x3=np.array(cv2.resize(frame3, (448, 320), interpolation = cv2.INTER_AREA))
                    # #x3=np.expand_dims(x3,axis=(2))
                    # x4=np.array(cv2.resize(frame4, (448, 320), interpolation = cv2.INTER_AREA))
                    
                    x1=cv2.resize(frame1, (448, 320), interpolation = cv2.INTER_AREA)
                    #x1=np.expand_dims(x1,axis=(2))
                    x2=cv2.resize(frame2, (448, 320), interpolation = cv2.INTER_AREA)
                    #x2=np.expand_dims(x2,axis=(2))
                    x3=cv2.resize(frame3, (448, 320), interpolation = cv2.INTER_AREA)
                    #x3=np.expand_dims(x3,axis=(2))
                    x4=cv2.resize(frame4, (448, 320), interpolation = cv2.INTER_AREA)
                    
                    #x4=np.expand_dims(x4,axis=(2))
                    #print(np.max(frame1,axis=(0,1)))
                    frames1[i,:,:]=x1
                    frames2[i,:,:]=x2
                    frames3[i,:,:]=x3
                    frames4[i,:,:]=x4

                    self.num_frame[i]=cap1.get(cv2.CAP_PROP_POS_FRAMES)

                    #print(np.min(frame_rgb_1),np.max(frame_rgb_1))
                    #print(type(frame_rgb_1))
                    
                    

                i+=1
                bar.update(1)
            
            frame_count+=1
       
        

          
        
        bar.close()

        print(" . . . . Predict Mask - Camera 1 . . . .")
        prediction1=tf.image.resize(self.model.predict(frames1),(1080,1920))[:,:,:,0].numpy()
        print(" . . . . Predict Mask - Camera 2 . . . .")
        prediction2=tf.image.resize(self.model.predict(frames2),(1080,1920))[:,:,:,0].numpy()
        print(" . . . . Predict Mask - Camera 3 . . . .")
        prediction3=tf.image.resize(self.model.predict(frames3),(1080,1920))[:,:,:,0].numpy()
        print(" . . . . Predict Mask - Camera 4 . . . .")
        prediction4=tf.image.resize(self.model.predict(frames4),(1080,1920))[:,:,:,0].numpy()
        
        print(". . . . Cumpute Homographie aah . . . ")
        
        self.mask_1_homo=np.array([cv2.warpPerspective(mask, self.H1, (1000, 1000)) for mask in tqdm(prediction1)],dtype=np.float32)
        self.mask_2_homo=np.array([cv2.warpPerspective(mask, self.H2, (1000, 1000)) for mask in tqdm(prediction2)],dtype=np.float32)
        self.mask_3_homo=np.array([cv2.warpPerspective(mask, self.H3, (1000, 1000)) for mask in tqdm(prediction3)],dtype=np.float32)
        self.mask_4_homo=np.array([cv2.warpPerspective(mask, self.H4, (1000, 1000)) for mask in tqdm(prediction4)],dtype=np.float32)

        
        mask_view=np.where(np.divide(np.sum([self.mask_1_homo,self.mask_2_homo,self.mask_3_homo,self.mask_4_homo],axis=0),4)>0.5,1,0)
        

        self.mask_view=np.array(mask_view,dtype=np.float32)


    def plot(self):
        fig, ax = plt.subplots()
        
        x=self.coordinate['x'].values
        y=self.coordinate['y'].values
        ax.scatter(x,y,c='lightcoral',zorder=2)
        ax.plot(x,y,c='silver',zorder=1)
        plt.xlim(0,1000)
        plt.ylim(0,1000)
        ax.add_patch(ptc.Rectangle((0,0),1000,1000,edgecolor ='red',fill=False,linewidth=7))
        ax.set_title(self.title_plot)
        plt.savefig(os.path.join(self.dir_coord_save,"trajectory.png"))
        #ax.invert_xaxis()
        #ax.axis('off')
        plt.show()
        

    def extract_trajectory(self):
        

        
            #fieldnames = ['x', 'y']
            
            #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            #writer.writeheader()
           

        back=cv2.imread(os.path.join("sample","homography","imgs","ref.png"))
        for i in range(len(self.mask_1_homo)) :
            mask_view_naz=self.mask_view[i].astype(np.uint8)

            contours, _ = cv2.findContours(image=mask_view_naz, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            max_contour=0
            for contour in contours:

                if cv2.contourArea(contour) >max_contour and  cv2.contourArea(contour) >500:
                    max_contour=cv2.contourArea(contour)

                    M=cv2.moments(contour)
            if max_contour>0:
                
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00']) 
                        
                    cv2.circle(back, (cx, cy), 7, (0, 0, 255), -1)
                    
                    self.coordinate=pd.concat([self.coordinate,pd.DataFrame({'x':[cx],'y':[cy],'num': self.num_frame[i]}) ]).reset_index(drop=True)
                    #writer.writerow({'x': cx, 'y': cy})

            #cv2.imshow('frame',frame_rgb_1)

            cv2.imshow('mask',back)
            cv2.imshow('mask_view',cv2.resize(self.mask_view[i], (300, 300), interpolation = cv2.INTER_AREA))
            cv2.imshow('mask1',cv2.resize(self.mask_1_homo[i], (250, 250), interpolation = cv2.INTER_AREA))
            cv2.imshow('mask2',cv2.resize(self.mask_2_homo[i], (250, 250), interpolation = cv2.INTER_AREA))
            cv2.imshow('mask3',cv2.resize(self.mask_3_homo[i], (250, 250), interpolation = cv2.INTER_AREA))
            cv2.imshow('mask4',cv2.resize(self.mask_4_homo[i], (250, 250), interpolation = cv2.INTER_AREA))
            #cv2.imshow('frame',self.frame[i])
            cv2.waitKey(100)

            
        
        
        cv2.destroyAllWindows()


    def test_delay(self):
        synchronize.delay_run_4(self.videos,self.video_delay,framestart=800)
        

    

