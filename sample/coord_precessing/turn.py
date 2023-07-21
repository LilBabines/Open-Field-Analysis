import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 


import sys 
sys.path.insert(1,"sample")
from smoother import w_e_smoother,mean_smooth
from coord_precessing import speed

def load_cvs(source):
    return pd.read_csv(source)

def get_slow_fast(x,y,t,l=100,d=1,seuil=40,windows=1):
    
    s,x,y=speed.speed(x,y,t,padding=windows,l=l)
    x_slow=[]
    y_slow=[]
    x_fast=[]
    y_fast=[]
    x_run=[]
    y_run=[]
    s_run=[]
    mean_speed_run=[]
    flag=0
    cpt=0
    for i in range(len(s)):
        if s[i]>seuil:
            if flag!=1:
                flag=1
            s_run.append(s[i])
            x_run.append(x[i])
            y_run.append(y[i])
        else :
            if flag!=0:
                flag=0
                if len(x_run)>=20:
                    x_fast.append(x_run)
                    y_fast.append(y_run)
                    mean_speed_run.append(np.mean(s_run))
                    cpt+=1
                x_run=[]
                y_run=[]
                s_run=[]
            x_slow.append(x[i])
            y_slow.append(y[i])
    print(f"{cpt} acélérations détectées")

    return x_slow,y_slow,x_fast,y_fast,mean_speed_run
def discr(u,v):
    return np.sign(v[1]*u[0] - u[1]*v[0])


def angle(u,v):

    
    return discr(u,v)*  np.arccos( np.dot(u,v)/   (np.linalg.norm(u)*np.linalg.norm(v) )  )*(180/np.pi) 
def vect(x1,x2,y1,y2):
    return np.array([x2-x1,y2-y1])




def run_curve(x,y):
    a=0
    for i in range(len(x)-2):
        u,v=vect(x[i+1],x[i],y[i+1],y[i]),vect(x[i+2],x[i+1],y[i+2],y[i+1])
        a+=angle(u,v)
    return a
