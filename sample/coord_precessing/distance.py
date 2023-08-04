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


def distance(dataframe,smooth=True,l=50,d=1):
    '''Calcule la distance parcourur par les rat, possibilité de smmoth'''
    x=dataframe['x'].values
    y=dataframe['y'].values
    t=dataframe['num'].values
    if smooth :

        x=w_e_smoother.whittaker_smooth(x,l,d=d)
        y=w_e_smoother.whittaker_smooth(y,l,d=d)
    
    time_travel=(t[-1]-t[0]) /60

    total_dist=0
    for i in range(len(x)-1):
        total_dist+=dist(x[i],x[i+1],y[i],y[i+1])
    print(f"Distance parcourue : {round(total_dist,2)}, en {round(time_travel,2)} secondes ")        #"
    

        
    return total_dist#,time_travel