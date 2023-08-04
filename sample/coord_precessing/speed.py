import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.insert(1,"sample")
from smoother import w_e_smoother

def miss_pts(source,frame_rate_analysis=2):
    '''regarde à l'aide des frames, combien de point manque à l'appel '''
    dataframe=load_cvs(source)
    t=np.array(dataframe['num'].values)
    cpt=0
    for i in range(len(t)-1):
        cpt+=((t[i+1]-t[i])//frame_rate_analysis )   -1
    print(f"{cpt} points manquants sur {len(t)} points acquis, soit {round(cpt/(len(t)+cpt) *100,2)} % des points")
    return cpt

def load_cvs(source):
    '''recupère un fichier csv comprenant des result de la trajectoire'''
    return pd.read_csv(source)

def dist(x1,x2,y1,y2):
    '''calcule la distance euclidienne entre deux point'''
    return np.linalg.norm(np.array([x1,y1])-np.array([x2,y2]))

def speed(x,y,t,padding,smooth=1,l=100,d=1):
    '''Calcule la vitesse en fonctoin du temps, padding correspond à lécart entre les point permettant de calculer la vitesse  
    smooth = 1 : smooth on x and y before cumpte speed
    smooth = 0 : no smooth
    smooth = 2 : smooth on speed 
    smooth= 3 : smmoth on x,y and speed'''
    if smooth in [1,3]:

        x=w_e_smoother.whittaker_smooth(x,l,d=d)
        y=w_e_smoother.whittaker_smooth(y,l,d=d)
    
    speed_= np.array([np.linalg.norm(np.array([x[i],y[i]])-np.array([x[i+padding],y[i+padding]]))/((t[i+padding]-t[i])/60) for i in range(len(x)-padding) ])
    
    if smooth in [2, 3]:

        speed_= w_e_smoother.whittaker_smooth(speed_,l,d=d)

        
    return speed_,x,y


def plot_speed(source,padding=2,smooth=True,d=2,l=100):
    dataframe=load_cvs(source)

    x=np.array(dataframe['x'].values)
    y=np.array(dataframe['y'].values)
    t=np.array(dataframe['num'].values)
    if smooth :

        x=w_e_smoother.whittaker_smooth(x,l,d=d)
        y=w_e_smoother.whittaker_smooth(y,l,d=d)

    
    s=speed(x,y,t,padding)[0]
    print(np.mean(s))
   
    plt.title(f" Vitesse en mm/s en foncction de la frame, padding = {padding}")
    plt.plot(w_e_smoother.whittaker_smooth(s,l,d=d))

    plt.show()



