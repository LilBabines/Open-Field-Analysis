import torch
import cv2
import numpy as np
from PIL import Image
import os

def load_model(dirYolo,source_pt):


    print(os.path.exists(source_pt))
    #Yolo=torch.load(yolo_weights_path)
    yolov5 = torch.hub.load(dirYolo, 'custom', source_pt, source='local')

    return yolov5

def score_frame(frame,modelYolo):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """
    
    #frame = [frame]
    results =modelYolo(frame)
    df=results.pandas().xyxyn[0]
    
    #labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    #return labels, cord
    cofidence=0

    cord=None
    label=None
    for _, row in df.iterrows():

        if row['confidence'] > cofidence :
            cord=[float(row['xmin']), float(row['ymin']),float(row['xmax']), float(row['ymax']),row['confidence']]
            label=row['name']
            

    return cord,label
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
    
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    if cord :
        #print(label)
        if label=='normal':
            #print('n',label)
            c=[0,0,255]
            conf=cord[4]
        else :
            c=[255,0,0]
            #print('e',label)
            conf=-cord[4]
        c=[255,0,0]
        x1, y1, x2, y2 = int(cord[0]*x_shape), int(cord[1]*y_shape), int(cord[2]*x_shape), int(cord[3]*y_shape)
        #bgr = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        #cv2.putText(frame, str(round(conf,2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
        
    return frame,conf
def test(path_to_video,frame_start,yolo_model,frame_rate_analysis=3):

    cap = cv2.VideoCapture(path_to_video)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    i=0

    if not cap.isOpened():
        print("Error opening video")

    while(cap.isOpened()):


        status, frame_rgb = cap.read()
        if status and i%frame_rate_analysis==0:
            cv2.imshow('ok',frame_rgb)
            cv2.imwrite('image.png',frame_rgb)

            img=Image.open('image.png')
            #print(np.asarray(img))

            score=score_frame(img,yolo_model)

            if score[1]!=None:
                #print(score[0])
                img=plot_boxes(score,frame_rgb)
            #else :
                #pass
                #print('no')
            cv2.imshow('yolo',frame_rgb)
            print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
        key = cv2.waitKey(1)
        
        i+=1
        if key == ord('a'):
            
            break
            
    cap.release()
    cv2.destroyAllWindows()