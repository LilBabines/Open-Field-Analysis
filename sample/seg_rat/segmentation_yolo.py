import torch
import cv2
import numpy as np
from PIL import Image
import os
import argparse

def load_model(source_pt=os.path.join(os.getcwd(),'model','poids','LAST_LAST_LAST.pt')):
    '''Charge le modèle de détection du rat.
    - Prédit la position du rat
    - Classifie sa posture : normal ou en redressement '''

    dirYolo=os.path.join(os.getcwd(),'model','yolov5')
    model = torch.hub.load(dirYolo, 'custom', path=source_pt, source='local')
    return model
  
def score_frame(frame,modelYolo):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of the most confidence box.
    """
    results =modelYolo(frame)
    df=results.pandas().xyxyn[0]
    cofidence=0
    cord=None
    label=None
    for _, row in df.iterrows():

        if row['confidence'] > cofidence :
            cord=[float(row['xmin']), float(row['ymin']),float(row['xmax']), float(row['ymax']),row['confidence']]
            label=row['name']
            cofidence=row['confidence']
    
    return cord,label

def mask(box,img_size=(1080,1920)):
    '''construit le mask correspondant à la box'''
    x_shape, y_shape = img_size[1], img_size[0]
    m=np.zeros(img_size)
    x1, y1, x2, y2 = int(box[0]*x_shape), int(box[1]*y_shape), int(box[2]*x_shape), int(box[3]*y_shape)

    m[y1:y2+1, x1:x2+1]=1
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
        if label=='normal':
            c=[0,0,255]
            conf=cord[4]
        else :
            c=[255,0,0]
            conf=-cord[4]
        
        x1, y1, x2, y2 = int(cord[0]*x_shape), int(cord[1]*y_shape), int(cord[2]*x_shape), int(cord[3]*y_shape)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2) 
    return frame,conf

def test(source,weights=None,frame_start=5000,frame_rate=3):
    '''test le fonctionnement du modèel sur un video
    - :param source: chemin de la video que va être analysée
    - :param frame_start: début de traitement'''

    assert  os.path.exists(source), f'vidéo introuvale, chemin spécifié : {source}'

    if weights :
        assert os.path.exists(weights), f'poids du modèle YOLOv5 est introuvable, chemin spécifié : {weights}'
        yolo_model=load_model(weights)

    else :
        yolo_model=load_model()

    cap = cv2.VideoCapture(source)


    if frame_start> cap.get(cv2.CAP_PROP_FRAME_COUNT)-1000:
        print(f'WARNING : frame start too hight, total frame number : {cap.get(cv2.CAP_PROP_FRAME_COUNT)} but frame start given : {frame_start}')
        frame_start=5000
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    i=0

    if not cap.isOpened():
        print("Error opening video")


    while(cap.isOpened()):


        status, frame_rgb = cap.read()
        if status and i%frame_rate==0:
            #cv2.imshow('ok',frame_rgb)
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
            #print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
        key = cv2.waitKey(1)
        
        i+=1
        if key == ord('a') or key==27:
            
            break
            
    cap.release()
    cv2.destroyAllWindows()

    i=0
    
    if not cap.isOpened():
        print("Error opening video")

    status, frame = cap.read()
    while(status):
        
        status, frame = cap.read()
        if status:
            frame=cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
            #frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
            #frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
            #frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
            #print(frame.shape)
            #frame=cv2.rotate(frame,cv2.ROTATE_180)
            #frame=cv2.flip(frame,1)
            cv2.imshow('frame', frame)
            
            output.write(frame)
            current_frame+=1
        key = cv2.waitKey(1)
        if key == ord('a'):
            print(current_frame)
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='.MP4 or .avi path file ',required=True)
    parser.add_argument('--weights', type=str, default=None, help='model path')

    parser.add_argument('--frame_start', type=int,default=5000, help='frame start analysis ')
    parser.add_argument('--frame_rate', type=int,default=3, help='frame rate analysis')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    test(**vars(opt))