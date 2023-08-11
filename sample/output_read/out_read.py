import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import argparse
import yaml
from matplotlib import cm
import numpy as np
import math
import sys
from matplotlib.colors import LogNorm
import seaborn as sns
import matplotlib.patches as mpatches

sys.path.insert(1,"sample")
from coord_precessing import distance,turn
from smoother import w_e_smoother
sys.path.insert(1,"utils")
import read_exel

with open('./cfg/run_cfg.yaml', 'r') as file :

    dict_cfg = yaml.safe_load(file)

    LESION_SIDE=dict_cfg['SIDE_LESION']

    ANALYSE_PATH=dict_cfg['ANALYSE_PATH']
    if not(os.path.exists(ANALYSE_PATH)):
        print("WARNING : dossier analyse pas trouvé, dossier créé, emplacement : {ANALYSE_PATH}")
        os.makedirs(ANALYSE_PATH)

    OUTPUT_PATH_PAW=dict_cfg['SAVE_PATH_PAW']
    assert os.path.exists(OUTPUT_PATH_PAW), f"dossier résultat appui {OUTPUT_PATH_PAW} pas trouvé"

    OUTPUT_PATH=dict_cfg['SAVE_PATH_TRAJECTORY']
    assert os.path.exists(OUTPUT_PATH), f"dossier résultat trajectoire {OUTPUT_PATH} pas trouvé"
    
    DATA_PATH = dict_cfg['DATA_PATH']

    assert os.path.exists(dict_cfg['DATA_PATH']), f" DATA dir doesn't exist, at {DATA_PATH} !! check documentary for set up the projet's hierachy"

    DATA_CONFIG = dict_cfg['DATA_CONFIG']
    assert os.path.exists(dict_cfg['DATA_CONFIG']) ,f" DATA configuration doesn't exist at {DATA_CONFIG} !! check documentary for set up the projet's hierachy"

    FRAME_RATE=dict_cfg['FRAME_RATE']

    MODEL_PATH=dict_cfg['MODEL_PATH']
    assert os.path.exists(dict_cfg['MODEL_PATH']) ,f" MODEL weights path doesn't exist at {MODEL_PATH} !! check documentary for set up the projet's hierachy"

    #print(set(LESION_SIDE.keys()),set(os.listdir(OUTPUT_PATH_PAW)))
    assert set(LESION_SIDE.keys())==set(os.listdir(OUTPUT_PATH_PAW)) ,f"Les coté de lésion ne corespondent pas au rat précent dans le dossier {OUTPUT_PATH_PAW}"

def get_path():
    path=os.path.join(OUTPUT_PATH,"config.xlsx")
    #print(path)
    assert os.path.exists(path)
    #print(os.path.exists(path))
    #print('Load', path)
    return path

def plot_HM_rat_1(rat,bin=100):
    f,axes=plt.subplots(2,2,num=rat+' - '+'Heat map',figsize=(13,13))#

    e=[]
    j=0

    for exp in ['pre_lesion','48_h_postL','1_semaine_postL','3_semaine_postL']:
        if exp in os.listdir(os.path.join(OUTPUT_PATH,rat)):
            #pritn(exp)
            e.append(exp)
            SAVE_PATH=os.path.join(OUTPUT_PATH,rat,exp)
            for i,_ in enumerate(os.listdir(SAVE_PATH)):
                dataframe=get_csv_of(rat,exp,num=i)
                axes[j//2][j%2].set_title(exp,fontsize = 14)
                sns.heatmap(data=bins(bin,dataframe['x'].values,dataframe['y'].values),xticklabels = False, yticklabels = False,cmap='inferno',norm=LogNorm(),cbar = False,ax=axes[j//2][j%2])
                axes[j//2][j%2].hlines(y=[10,90],xmin=10,xmax=90,linewidth=3,color='red', linestyle = "dashed")
                axes[j//2][j%2].vlines(x=[10,90],ymin=10,ymax=90,color='red',linewidth=3, linestyle = "dashed")
                j+=1
                    
                
    #f.tight_layout(pad=1.0)
    plt.show()

    #plt.legend(loc='upper left', labels=e)

def plot_HM_rat_2(rat,bin=100,red=True):
    f,axes=plt.subplots(2,2,num=rat+' - '+'Heat map',figsize=(13,13))#

    e=[]
    j=0

    for exp in ['pre_lesion','1_semaine_postL','3_semaine_postL']:
        if exp in os.listdir(os.path.join(OUTPUT_PATH,rat)):
            #pritn(exp)
            e.append(exp)
            SAVE_PATH=os.path.join(OUTPUT_PATH,rat,exp)
            for i,_ in enumerate(os.listdir(SAVE_PATH)):
                dataframe=get_csv_of(rat,exp,num=i)
                axes[j//2][j%2].set_title(exp,fontsize = 14)
                sns.heatmap(data=bins2(bin,dataframe['x'].values,dataframe['y'].values),xticklabels = False, yticklabels = False,cmap='inferno',norm=LogNorm(),cbar = False,ax=axes[j//2][j%2])#
                
                if red:
                    axes[j//2][j%2].hlines(y=20,xmin=20,xmax=100,linewidth=5,color='red', linestyle = "dashed")
                    axes[j//2][j%2].vlines(x=20,ymin=20,ymax=100,linewidth=5,color='red', linestyle = "dashed")
                j+=1
        
    #f.tight_layout(pad=1.0)
    plt.show()
    
def plot_HM(dataframe,red=True):
    f, axes = plt.subplots(1, 2,figsize=(20,10))
    sns.heatmap(data=bins2(100,dataframe['x'].values,dataframe['y'].values),xticklabels = False, yticklabels = False,cmap='inferno',norm=LogNorm(),ax=axes[0],cbar = False) 
    
    sns.heatmap(data=bins(100,dataframe['x'].values,dataframe['y'].values),xticklabels = False, yticklabels = False,cmap='inferno',norm=LogNorm(),ax=axes[1],cbar = False) 
    if red:
        axes[0].hlines(y=20,xmin=20,xmax=100,linewidth=5,color='red', linestyle = "dashed")
        axes[0].vlines(x=20,ymin=20,ymax=100,linewidth=5,color='red', linestyle = "dashed")
        axes[1].hlines(y=[10,90],xmin=10,xmax=90,linewidth=5,color='red', linestyle = "dashed")
        axes[1].vlines(x=[10,90],ymin=10,ymax=90,color='red',linewidth=5, linestyle = "dashed")

def bins(bin,x,y):
    shape = [int(bin),int(bin)]
    bins = np.zeros(shape, dtype=int) +1

    xind = x // (1000/bin)

    yind = y // (1000/bin)

    for ind in zip(yind, xind):
        #print(ind)

        bins[int(ind[0]),int(ind[1])] += 1
    return bins

def bins2(bin,x,y):
    shape = [int(bin),int(bin)]
    bins = np.ones(shape, dtype=int)

    X=np.where(x>500,1000-x,x)
    Y=np.where(y>500,1000-y,y)

    xind = X //  (500/bin)

    yind = Y // (500/bin)

    for x1,y1 in zip(yind, xind):
        #print(ind)
        

        bins[int(x1),int(y1)] += 1
    return bins

def get_csv(rat,exp,num=0):
    dataframe=pd.read_csv(os.path.join(OUTPUT_PATH,rat,exp,"coordinates"+str(num),'coord.csv'))
    return dataframe

def get_csv_of(rat,exp,num=0,clean=True):
    #print(rat,exp,'bug')
    fin = end_OF(rat,exp,num=num)
    dataframe=pd.read_csv(os.path.join(OUTPUT_PATH,rat,exp,"coordinates"+str(num),'coord.csv'))

    df=dataframe[dataframe['num']<fin]
    s=start(df)
    frame=dataframe['num'].values[s]
    #print(df.sample(20))
    #print(s,fin)
    df=df[df['num']>=frame]

    return df 

def get_csv_paw(rat,exp,num=0,clean=True):
    #print(rat,exp,'bug')
    #fin = end_OF(rat,exp,num=num)
    print(OUTPUT_PATH_PAW)
    dataframe=pd.read_csv(os.path.join(OUTPUT_PATH_PAW,rat,exp,"coordinates"+str(num),'coord.csv'))

    


    return dataframe

def group_rat():
    df=None
    f=True
    for rat in os.listdir(OUTPUT_PATH)[1:]:

        for exp in ['pre_lesion','1_semaine_postL','3_semaine_postL']:
            
            if exp in os.listdir(os.path.join(OUTPUT_PATH,rat)):

                print(rat,exp)

                SAVE_PATH=os.path.join(OUTPUT_PATH,rat,exp)

                for name in enumerate(os.listdir(SAVE_PATH)):
                    #dataframe=get_csv(rat,exp,num=i)
                    i=int(name[-1])
                    print(i)
                    dataframe=get_csv_of(rat,exp,num=i)
                    dataframe['rat']=rat
                    dataframe['exp']=exp
                    if f:
                        df =dataframe
                        f=False
                    else: 
                        df=pd.concat([df,dataframe],ignore_index=True)
    return df

def group_rat_paw():
    df=None
    f=True
    for rat in os.listdir(OUTPUT_PATH_PAW):

        for exp in ['pre_lesion','1_semaine_postL','3_semaine_postL']:
            
            if exp in os.listdir(os.path.join(OUTPUT_PATH_PAW,rat)):

                

                SAVE_PATH=os.path.join(OUTPUT_PATH_PAW,rat,exp)

                for name in os.listdir(SAVE_PATH):
                    #dataframe=get_csv(rat,exp,num=i)
                    print(rat,exp,name)
                    i=int(name[-1])

                    dataframe=get_csv_paw(rat,exp,num=i)
                    dataframe['rat']=rat
                    dataframe['exp']=exp
                    if f:
                        df =dataframe
                        f=False
                    else: 
                        df=pd.concat([df,dataframe],ignore_index=True)
    return df
               
def clean(dataframe):
    rat=dataframe.loc[ (dataframe['mask_score']>0.8)].reset_index()
    #_,index=miss_cam(rat)


    #x,y,t=np.delete(rat['x'].values,index),np.delete(rat['y'].values,index),np.delete(rat['num'].values,index)
    return rat['x'].values,rat['y'].values,rat['num'].values

def get_trajectory(dataframe):
    x=np.array(dataframe['x'].values)
    y=np.array(dataframe['y'].values)
    return x,y

def get_num_frame(dataframe):
    t=np.array(dataframe['num'].values)
    return t

def plot_trajectory(X,Y,rat,exp,num):

        _, ax = plt.subplots()
        ax.scatter(X,Y,c='lightcoral',zorder=2)
        ax.plot(X,Y,c='silver',zorder=1)
        plt.xlim(0,1000)
        plt.ylim(0,1000)
        ax.add_patch(ptc.Rectangle((0,0),1000,1000,edgecolor ='red',fill=False,linewidth=7))
        ax.set_title(rat + exp )
        #plt.savefig(os.path.join(OUTPUT_PATH,rat,exp,f"trajectory{num}.png"))
        #ax.invert_xaxis()
        #ax.axis('off')
        plt.show()

def get_label(dataframe):
    return np.array(dataframe['label'].values)

def end_OF(rat,exp,num=0):
    

    conf=read_exel.get_df(cfg=DATA_CONFIG)
  
        
    #print(conf)
    #print(rat,exp)
    rows=conf.loc[(conf['rat']==rat) & (conf['exp']==exp) ]
    #print(rows)
    if math.isnan(rows['end_frame'].values[num]):
        time=rows['end_time'].values[num]
        return (time.minute+time.hour*60 )  *60
    else :
        return rows['end_frame'].values[num]

def diff(rat):
    x=rat['x'].values
    y=rat['y'].values
    l=np.array([])
    for i in range(len(x)-1):

        l=np.append(l,np.abs(x[i]-x[i+1])+np.abs(y[i]-y[i+1]))  
    return l

def glis(x):
    c=0
    while np.max(x[c:c+50]) >80:
        c+=1
    return c

def start(rat):
    return glis(diff(rat))

def miss_pts(dataframe):
    t=np.array(dataframe['num'].values)
    start=t[0]
    cpt=0
    for i in range(len(t)-1):
        
        cpt+=((t[i+1]-t[i])//2 )   -1
        if (t[i+1]-t[i]) >100:
            #print(t[i])
            start=t[i+1]
            cpt=0

    #print(f"{cpt} points manquants sur {len(t)} points acquis, soit {round(cpt/(len(t)+cpt) *100,2)} % des points, start {start}")
    return cpt,start

def miss_cam(dataframe):
    conf=dataframe['cam_confidence'].values
    cpt=[0,0,0,0]
    index=[]
    #print(type(conf[0]))
    for j,x in enumerate(conf):
        #print(x)
        for i,l in enumerate(x[1:-1].split(',')):
            
            if float(l)==0:
                
                #print(i)
                cpt[i]+=1
                index.append(j)
    return cpt,np.unique(index)

def time_video(rat):

    return rat['num'].values[-1]-rat['num'].values[0]

def time_bord(x,y,num):
    time_bord=0
    time_center=0
    for i in range(1,len(x)):
        if x[i]>250 and x[i]<750 and y[i]<750 and y[i]>250:
            time_center+=num[i]-num[i-1]
        else :
            time_bord+=num[i]-num[i-1]

    return time_bord/60,time_center/60

def distance_bord(X,Y):
    dist=[]
    for x,y in zip(X,Y):
        #dist.append(np.sqrt( (500-x)**2 + (500-y)**2))s
        dist.append( np.abs(500-x) + np.abs(500-y)) 
    return dist

def distance_bord2(X,Y):
    dist=[]
    for u,v in zip(X,Y):
        #dist.append(np.sqrt( (500-x)**2 + (500-y)**2))s
        dist.append( min(min(u,1000-u),min(v,1000-v))) 
    return dist

def plot_distrib_bord():
    df=group_rat()
    n=len(df['exp'].unique())
    assert n==3
    sns.set_theme()
    fig, axs = plt.subplots(figsize=(20, 10))
    axs.set_title(f'Répartition de la Distance au Dord de tout les rats')
    axs.set_xlabel('distance en mm du bord')
    i=0
    for exp in df['exp'].unique():
        data=df[df['exp']==exp]
        X=data['x']
        Y=data['y']
        hist=distance_bord2(X,Y)


        sns.kdeplot(x=hist,ax=axs,label=exp)
        i+=1
    fig.legend()

def plot_bord(dataframe):
    sns.set_theme()
    sns.histplot(x=distance_bord(dataframe['x'].values,dataframe['y'].values), kde=True,bins=60)

def plot_bord_rat(rat):
    plt.subplots(num=rat+' - '+'Répartition distance au bord',figsize=(15,15))

    e=[]
    for exp in ['pre_lesion','48_h_postL','1_semaine_postL','3_semaine_postL']:
        if exp in os.listdir(os.path.join(OUTPUT_PATH,rat)):
            e.append(exp)
            SAVE_PATH=os.path.join(OUTPUT_PATH,rat,exp)
            for i,_ in enumerate(os.listdir(SAVE_PATH)):
                dataframe=get_csv_of(rat,exp,num=i)
                plot_bord(dataframe)
    plt.legend(loc='upper left', labels=e)
    plt.show()

def elevation_time(t,label):
    cpt=0
    for i in range(1,len(t)):
        if label[i-1]==label[i]==1 or label[i-1]==label[i]=='elevation':
            cpt+=t[i]-t[i-1]
    return cpt

def smooth_elevation2(dataframe,frame_mean=11):
    
    cam_confidence=np.array(dataframe['cam_confidence'].values)
    x=np.array(dataframe['num'].values)
    conf=np.mean(np.array([np.array(i[1:-1].split(","),dtype=object).astype(float) for i in cam_confidence] ,dtype=object),axis=1)
    label= [2*(np.mean(conf[i-frame_mean//2:i+frame_mean//2])<0)-1 for i in range(frame_mean//2,len(conf)-frame_mean//2 ) ] 
    label=[label[0]]*(frame_mean//2)  + label +[label[-1]]*(frame_mean//2)
    
    return x,label

def smooth_elevation(dataframe,frame_mean=11):
    
    cam_confidence=np.array(dataframe['cam_confidence'].values)
    x=np.array(dataframe['num'].values)
    conf=np.mean(np.array([np.array(i[1:-1].split(","),dtype=object).astype(float) for i in cam_confidence] ,dtype=object),axis=1)
    label= [2*(np.mean(conf[i-frame_mean//2:i+frame_mean//2])<0)-1 for i in range(frame_mean//2,len(conf)-frame_mean//2 ) ] 
    label=[label[0]]*(frame_mean//2)  + label +[label[-1]]*(frame_mean//2)
    
    return x,label

def count_num(t,label):
    flag=label[0]
    lab=[]

    cpt=0
    for i in range(1,len(label)):
        if label[i]==flag:
            cpt+=t[i]-t[i-1]
        else :
            if flag==1 or flag=='elevation':
                lab.append(t[i])
            flag=label[i]
            cpt=0
    return lab

def count(t,label):
    flag=label[0]
    lab=[]
    cpt=0
    for i in range(1,len(label)):
        if label[i]==flag:
            cpt+=t[i]-t[i-1]
        else :
            if flag==1 or flag=='elevation':
                lab.append(cpt)
            flag=label[i]
            cpt=0
    return lab

def global_tab():
    cpt1=np.array([0,0,0,0])
    total_tab=pd.DataFrame(columns=('rat','exp','ele_count','ele_time','ele_norm','distance','number_run','speed_run','total_frame','miss_pts','miss_cam','time_bord','time_center'))
    #print(os.listdir(OUTPUT_PATH)[1:],type(os.listdir(OUTPUT_PATH)))

    
    for rat in os.listdir(OUTPUT_PATH):
        for exp in ['pre_lesion','1_semaine_postL','3_semaine_postL']:
            

            if exp in os.listdir(os.path.join(OUTPUT_PATH,rat)):


            
                print(rat,exp)
                SAVE_PATH=os.path.join(OUTPUT_PATH,rat,exp)

                for i,name in enumerate(os.listdir(SAVE_PATH)):
                    dataframe=get_csv(rat,exp,num=i)
                    deb=start(dataframe)
                    
                    dataframe=get_csv_of(rat,exp,num=i)
                    cpt,_=miss_pts(dataframe)
                    cpt_cam,_=miss_cam(dataframe)
                    totalF=time_video(dataframe)
                    
                    x,label=smooth_elevation(dataframe,13)
                    x,label2=smooth_elevation(dataframe,17)
                    x,label3=smooth_elevation(dataframe,21)
                    
                    cpt1=cpt1+np.array(cpt_cam)
                    count1,count2,count3=count(x,label),count(x,label2),count(x,label3)
                    time_ele=elevation_time(x,label)

                    time_ele2=elevation_time(x,label2)
                    time_ele3=elevation_time(x,label3)

                    dist=distance.distance(dataframe)
                    t_bord,t_center=time_bord(dataframe['x'].values,dataframe['y'].values,dataframe['num'].values)
                    
                    x_slow,y_slow,x_fast,y_fast,mean_run_spe=turn.get_slow_fast(dataframe['x'].values,dataframe['y'].values,dataframe['num'].values,seuil=35,windows=1)
                    
                    total_tab.loc[len(total_tab)]={'rat':rat,'exp':exp,'ele_count':np.mean([len(count1),len(count2),len(count3)])/(totalF/60**2),'ele_time':100*np.mean([time_ele,time_ele2,time_ele3])/totalF,'distance':60*dist/totalF,'number_run':len(mean_run_spe)/(totalF/60**2),'speed_run':np.mean(mean_run_spe),'total_frame':totalF,'ele_norm':np.mean([np.mean(count1),np.mean(count2),np.mean(count3)])/60,'time_bord':100*t_bord/(totalF/60),'time_center':100*t_center/(totalF/60),'miss_pts':cpt,'miss_cam':cpt_cam}         
    total_tab.to_csv(os.path.join(ANALYSE_PATH,'resultats10.csv'))   

def plot_elevation(dataframe,rat,exp):
    
    x,label=smooth_elevation(dataframe,frame_mean=15)
    c=count(x,label)
    print(len(c))
    fig, axs = plt.subplots(2)
    fig.suptitle(rat +' '+exp)
    axs[0].hist(c,bins=50)
    axs[0].set_title("Histogramme durée d'élévation")
    axs[1].plot(x,label)
    axs[1].set_title('Evolution de la posture en fonction du numéro de frame')
    plt.show()
    return fig,axs

def rat_stat(rat,num=0):

    stats=pd.read_csv(f'resultats{num}.csv')
    row=stats.loc[ stats['rat']==rat]

    return row

def get_stat(num=6,shamm=False):
    df=pd.read_csv(f'resultats{num}.csv')
    if not shamm:
        df=df[df['rat']!='P1V']

    #print(df)
    return df

def plot_rat_stat(row):
    fig, axs = plt.subplots(6,num=(row['rat'].values[0]),figsize=(8, 10))
    #fig.canvas.set_window_title(row['rat'].values[0])
    fig.suptitle(row['rat'].values[0] )

    

    n=len(row['ele_count'].values)+1
    x=range(1,n)
    my_xticks = row['exp'].values
    axs[0].set_xticks(x, my_xticks)
    axs[1].set_xticks(x, my_xticks)
    axs[2].set_xticks(x, my_xticks)
    axs[3].set_xticks(x, my_xticks)
    axs[4].set_xticks(x, my_xticks)
    axs[5].set_xticks(x, my_xticks)

    axs[0].plot(x,row['ele_count'].values,markersize=8,marker="x")
    axs[0].set_title("Nombre moyen d'élévation par minute")

    axs[1].plot(x,row['ele_time'].values,markersize=8,marker="x")
    axs[1].set_title('Pourcentage du temps resté en élévation')

    axs[2].plot(x,row['ele_norm'].values,markersize=8,marker="x")
    axs[2].set_title("Temps moyen par élévation")

    axs[3].plot(x,row['distance'].values,markersize=8,marker="x")
    axs[3].set_title("Vitesse moyenne")

    axs[4].plot(x,row['number_run'].values,markersize=8,marker="x")
    axs[4].set_title('Nombre moyen de déplacement par minute')

    axs[5].plot(x,row['speed_run'].values,markersize=8,marker="x")
    axs[5].set_title("Vitesse moyenne des déplacements")

    fig.tight_layout(pad=1.0)

    #plt.show()

def add_shamm(fig1,fig2,axs1,axs2):
    stat=get_stat(shamm=True)
    row=stat.loc[stat['rat']=='P1V']
    #print(row)
    #x=range(0,3)

    sns.lineplot(data=row,x="exp",y='ele_count',ax=axs1[0])#,palette=palette,order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],dodge=False
    #axs1[0].plot(x,row['ele_count'].values,markersize=8,marker="x",c='red')
    #axs[0].set_title("Nombre moyen d'élévation par minute")
    sns.lineplot(data=row,x="exp",y='ele_time',ax=axs1[1])#,order=["pre_lesion",'1_semaine_postL','3_semaine_postL']
    #axs1[1].plot(x,row['ele_time'].values,markersize=8,marker="x",c='red')
    #axs[1].set_title('Pourcentage du temps resté en élévation')

    sns.lineplot(data=row,x="exp",y='ele_norm',ax=axs1[2])#
    #axs1[2].plot(x,row['ele_norm'].values,markersize=8,marker="x",c='red')
    #axs[2].set_title("Temps moyen par élévation")

    sns.lineplot(data=row,x="exp",y='distance',ax=axs2[0])#
    #axs2[0].plot(x,row['distance'].values,markersize=8,marker="x",c='red')
    #axs[3].set_title("Vitesse moyenne")

    sns.lineplot(data=row,x="exp",y='number_run',ax=axs2[1])#
    #axs2[1].plot(x,row['number_run'].values,markersize=8,marker="x",c='red')
    #axs[4].set_title('Nombre moyen de déplacement par minute')

    sns.lineplot(data=row,x="exp",y='speed_run',ax=axs2[2])#
    #axs[5].set_title("Vitesse moyenne des déplacements")

def plot_stat2(data,sham=True):
    sns.set_theme()
    
    fig1, axs1 = plt.subplots(3,figsize=(8, 10))
    
    axs1[0].set_title("Nombre moyen d'élévations par minute")
    palette = sns.color_palette(["#4c72b0","#c44e52","#55a868"])
    
    
    sns.stripplot(data=data,x="exp",y='ele_count',ax=axs1[0],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],dodge=False, jitter=False,palette=palette,size=7,linewidth=1)
    sns.boxplot(data=data,x="exp",y='ele_count',ax=axs1[0],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],boxprops={'alpha': 0.6},palette=palette)
    axs1[0].set_ylabel("Nombre d'élévations")
    axs1[0].set_xlabel("Expérience")
    

    axs1[1].set_title("Pourcentage du temps resté en élévation")
    sns.boxplot(data=data,x="exp",y='ele_time',ax=axs1[1],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],boxprops={'alpha': 0.6},palette=palette)#,hue='ele_time'
    sns.stripplot(data=data,x="exp",y='ele_time',ax=axs1[1],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],dodge=False, jitter=False,palette=palette,size=7,linewidth=1)
    axs1[1].set_ylabel("Temps en pourcentage")
    axs1[1].set_xlabel("Expérience")
    
    
   
    axs1[2].set_title("Temps moyen par élévation")
    sns.boxplot(data=data,x="exp",y='ele_norm',ax=axs1[2],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],boxprops={'alpha': 0.6},palette=palette)
    sns.stripplot(data=data,x="exp",y='ele_norm',ax=axs1[2],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],dodge=False, jitter=False,palette=palette,size=7,linewidth=1)
    axs1[2].set_ylabel("Temps en seconde")
    axs1[2].set_xlabel("Expérience")
    

    fig2, axs2 = plt.subplots(3,figsize=(8, 10))
    
    axs2[0].set_title("Distance moyenne parcourue en une seconde")
    sns.boxplot(data=data,x="exp",y='distance',ax=axs2[0],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],boxprops={'alpha': 0.6},palette=palette)#,hue='ele_time'
    sns.stripplot(data=data,x="exp",y='distance',ax=axs2[0],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],dodge=False, jitter=False,palette=palette,size=7,linewidth=1)
    axs2[0].set_ylabel("Distance en mm")
    axs2[0].set_ylabel("Expérience")
    

    axs2[1].set_title("Nombre moyen de déplacements par minute")
    sns.boxplot(data=data,x="exp",y='number_run',ax=axs2[1],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],boxprops={'alpha': 0.6},palette=palette)
    sns.stripplot(data=data,x="exp",y='number_run',ax=axs2[1],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],dodge=False, jitter=False,palette=palette,size=7,linewidth=1)
    axs2[1].set_ylabel("Nombre de déplacements")
    axs2[1].set_xlabel("Expérience")
    

    axs2[2].set_title("Vitesse moyenne des déplacements")
    sns.boxplot(data=data,x="exp",y='speed_run',ax=axs2[2],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],boxprops={'alpha': 0.6},palette=palette)
    sns.stripplot(data=data,x="exp",y='speed_run',ax=axs2[2],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],dodge=False, jitter=False,palette=palette,size=7,linewidth=1)
    axs2[2].set_ylabel("Vistess en mm/s ")
    axs2[2].set_xlabel("Expérience")
    
    if sham:
        stat=get_stat(shamm=True)
        row=stat.loc[stat['rat']=='P1V']
        #print(row)
        #x=range(0,3)

        sns.lineplot(data=row,x="exp",y='ele_count',ax=axs1[0],c='purple',markers=True,linewidth=3)#,palette=palette,order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],dodge=False
        #axs1[0].plot(x,row['ele_count'].values,markersize=8,marker="x",c='red')
        #axs[0].set_title("Nombre moyen d'élévation par minute")
        sns.lineplot(data=row,x="exp",y='ele_time',ax=axs1[1],c='purple',markers=True,linewidth=3)#,order=["pre_lesion",'1_semaine_postL','3_semaine_postL']
        #axs1[1].plot(x,row['ele_time'].values,markersize=8,marker="x",c='red')
        #axs[1].set_title('Pourcentage du temps resté en élévation')

        sns.lineplot(data=row,x="exp",y='ele_norm',ax=axs1[2],c='purple',markers=True,linewidth=3)#
        #axs1[2].plot(x,row['ele_norm'].values,markersize=8,marker="x",c='red')
        #axs[2].set_title("Temps moyen par élévation")

        sns.lineplot(data=row,x="exp",y='distance',ax=axs2[0],c='purple',markers=True,linewidth=3)#
        #axs2[0].plot(x,row['distance'].values,markersize=8,marker="x",c='red')
        #axs[3].set_title("Vitesse moyenne")

        sns.lineplot(data=row,x="exp",y='number_run',ax=axs2[1],c='purple',markers=True,linewidth=3)#
        #axs2[1].plot(x,row['number_run'].values,markersize=8,marker="x",c='red')
        #axs[4].set_title('Nombre moyen de déplacement par minute')

        sns.lineplot(data=row,x="exp",y='speed_run',ax=axs2[2],c='purple',markers=True,linewidth=3)#
    axs2[0].set_ylim(0, None)
    axs2[2].set_ylim(0, None)
    axs2[1].set_ylim(0, None)
    axs1[2].set_ylim(0, None)
    axs1[1].set_ylim(0, None)
    axs1[0].set_ylim(0, None)
    axs2[0].set_xlabel("Expérience")
    axs2[2].set_xlabel("Expérience")
    axs2[1].set_xlabel("Expérience")
    axs1[2].set_xlabel("Expérience")
    axs1[1].set_xlabel("Expérience")
    axs1[0].set_xlabel("Expérience")
    
    fig1.tight_layout()
    fig2.tight_layout()
    #return fig1,fig2,axs1,axs2

def plot_stat(data):
    
    #cmap =cm.get_cmap('hsv', 11)
    #colors=cmap(np.arange(0,10))
    #print( colors )
    
    title=["Nombre moyen d'élévation par minute",'Pourcentage du temps resté en élévation',"Temps moyen par élévation",'Distance parcouru','Nombre moyen de déplacement par minute','Vitesse moyenne des déplacements']
    #print(len(cmap.colors))
    #array=cmap.get_array()
    #plt.figure(figsize=(50,50))
    fig1, axs1 = plt.subplots(3,figsize=(8,10))
    fig2, axs2 = plt.subplots(3,figsize=(8,10))
    axs1[0].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    axs1[1].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    axs1[2].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    axs2[0].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    axs2[1].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    axs2[2].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    for j,S in enumerate(['ele_count','ele_time','ele_norm','distance','number_run','speed_run']):
        cpt=0
        
        for rat in data['rat'].unique() :

            stat=data.loc[ data['rat']==rat].reset_index()
            n=len(stat['ele_count'].values)+1
            x=[]
            y=[]
            
            for i,exp in enumerate(['pre_lesion','1_semaine_postL','3_semaine_postL']):
                

                if exp in stat['exp'].values:
                    
                    vals=stat.loc[ stat['exp']==exp][S].values
                    y+=list(vals)
                    x+=[i+1]*len(vals)
                
           # print(x,y)
            color='blue'
            
            if rat=='P1V':
                color='red'
                

            
            if j >2:

                axs2[j-3].set_title(title[j])
                if j==0:

                    axs2[j-3].plot(x,y,c=color,linewidth=3,zorder=1)
                else :
                    axs2[j-3].plot(x,y,c=color,linewidth=3,zorder=1)
                cpt+=1
            else:
                axs1[j].set_title(title[j])
                if j==0:

                    axs1[j].plot(x,y,c=color,linewidth=3,zorder=1)
                else :
                    axs1[j].plot(x,y,c=color,linewidth=3,zorder=1)
                cpt+=1

    for j,S in enumerate(['ele_count','ele_time','ele_norm','distance','number_run','speed_run']):
        if j>2:
            count=[data.loc[data['exp']=='pre_lesion'][S].values,data.loc[data['exp']=='1_semaine_postL'][S].values,data.loc[data['exp']=='3_semaine_postL'][S].values]
            for i,stat in enumerate(count):
                axs2[j-3].scatter([i+1]*len(stat),stat ,c='salmon' ,zorder=2)
        else:

            count=[data.loc[data['exp']=='pre_lesion'][S].values,data.loc[data['exp']=='1_semaine_postL'][S].values,data.loc[data['exp']=='3_semaine_postL'][S].values]
            for i,stat in enumerate(count):
                axs1[j].scatter([i+1]*len(stat),stat ,c='salmon' ,zorder=2)
    blue_patch = mpatches.Patch(color='blue', label='implanté')
    red_patch = mpatches.Patch(color='red', label='contrôle')
    
    axs1[0].set_ylabel("Nombre d'élévation")
    
    axs1[1].set_ylabel("Temps en pourcentage")
   
    axs1[2].set_ylabel("Temps en seconde")
    
    axs2[0].set_ylabel("Distance en mm")
   
    axs2[1].set_ylabel("Nombre de déplacement")
   
    axs2[2].set_ylabel("Vistess en mm/s ")
    
    fig1.legend(handles=[blue_patch,red_patch])
    fig2.legend(handles=[blue_patch,red_patch])

def plot_stat1(data,norm=True):

    sns.set_theme()
    cmap =cm.get_cmap('hsv', 11)
    #colors=cmap(np.arange(0,10))
    #print( colors 
    colorS=sns.color_palette("tab10",as_cmap=True).colors
    #print(type(colors))
    #print(type(colors2))
    title=["Nombre moyen d'élévations par minute",'Pourcentage du temps resté en élévation',"Temps moyen par élévation",'Distance parcourue','Nombre moyen de déplacements par minute','Vitesse moyenne des déplacements']
    #print(len(cmap.colors))
    #array=cmap.get_array()
    #plt.figure(figsize=(50,50))
    fig1, axs1 = plt.subplots(3,figsize=(8,10))
    fig2, axs2 = plt.subplots(3,figsize=(8,10))
    axs1[0].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    axs1[1].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    axs1[2].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    axs2[0].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    axs2[1].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    axs2[2].set_xticks([1,2,3], ['Pré-lésion','1 semaine post lésion','3 semaine post-lésion'])
    for j,S in enumerate(['ele_count','ele_time','ele_norm','distance','number_run','speed_run']):
        cpt=0
        
        for rat in data['rat'].unique() :
            if rat =='P1V':
                col='darkj'
            stat=data.loc[ data['rat']==rat].reset_index()
            n=len(stat['ele_count'].values)+1
            x=[]
            y=[]
            
            for i,exp in enumerate(['pre_lesion','1_semaine_postL','3_semaine_postL']):
                

                if exp in stat['exp'].values:
                    
                    vals=stat.loc[ stat['exp']==exp][S].values
                    y+=list(vals)
                    x+=[i+1]*len(vals)
                
           # print(x,y)
            if norm :

                y=y/y[0]
            
            if j >2:

                axs2[j-3].set_title(title[j])
                if j==3:

                    axs2[j-3].plot(x,y,c=colorS[cpt],linewidth=2,zorder=1,label=rat)
                else :
                    axs2[j-3].plot(x,y,c=colorS[cpt],linewidth=2,zorder=1)
                cpt+=1
            else:
                axs1[j].set_title(title[j])
                if j==0:

                    axs1[j].plot(x,y,c=colorS[cpt],linewidth=2,zorder=1,label=rat)
                else :
                    axs1[j].plot(x,y,c=colorS[cpt],linewidth=2,zorder=1)
                cpt+=1

    #for j,S in enumerate(['ele_count','ele_time','ele_norm','distance','number_run','speed_run']):
        #if j>2:
            #count=[data.loc[data['exp']=='pre_lesion'][S].values, data.loc[data['exp']=='48_h_postL'][S].values,data.loc[data['exp']=='1_semaine_postL'][S].values,data.loc[data['exp']=='3_semaine_postL'][S].values]
            #for i,stat in enumerate(count):
                #axs2[j-3].scatter([i+1]*len(stat),stat ,c='salmon' ,zorder=2)
        #else:

            #count=[data.loc[data['exp']=='pre_lesion'][S].values, data.loc[data['exp']=='48_h_postL'][S].values,data.loc[data['exp']=='1_semaine_postL'][S].values,data.loc[data['exp']=='3_semaine_postL'][S].values]
           # for i,stat in enumerate(count):
                #axs1[j].scatter([i+1]*len(stat),stat ,c='salmon' ,zorder=2)
    
    axs1[0].set_ylabel("Nombre d'élévations")
    
    axs1[1].set_ylabel("Temps en pourcentage")
   
    axs1[2].set_ylabel("Temps en seconde")
    
    axs2[0].set_ylabel("Distance en mm")
   
    axs2[1].set_ylabel("Nombre de déplacement")
   
    axs2[2].set_ylabel("Vistess en mm/s ")
    fig1.legend()
    fig2.legend()
    axs2[0].set_ylim(0, None)
    axs2[2].set_ylim(0, None)
    axs2[1].set_ylim(0, None)
    axs1[2].set_ylim(0, None)
    axs1[1].set_ylim(0, None)
    axs1[0].set_ylim(0, None)
    fig1.tight_layout()
    fig2.tight_layout()

def plot_bord_time():
    data=get_stat(num=10)
    sns.set_theme()
    fig1, axs1 = plt.subplots(1,figsize=(8, 10))
    #sns.set_theme()
    stat=get_stat(shamm=True,num=10)
    row=stat.loc[stat['rat']=='P1V']
    #print(row)
    x=range(0,3)
    palette = sns.color_palette(["#4c72b0","#c44e52","#55a868"])
    axs1.set_title("Pourcentage de temps passé au centre")
    sns.boxplot(data=data,x="exp",y='time_center',ax=axs1,order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],boxprops={'alpha': 0.6},palette=palette)#,hue='ele_time'
    sns.stripplot(data=data,x="exp",y='time_center',ax=axs1,order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],dodge=False, jitter=False,palette=palette,size=7)
    axs1.set_ylabel("Temps en pourcentage")
    axs1.set_xlabel(None)

    
    axs1.plot(x,row['time_center'].values,markersize=8,marker="x",c='red')


    #axs1.set_ylim(0, None)
    # axs1[1].set_title("Pourcentage de temps passé au centre")
    # sns.boxplot(data=data,x="exp",y='time_center',ax=axs1[1],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],boxprops={'alpha': 0.6},palette=palette)#,hue='ele_time'
    # sns.stripplot(data=data,x="exp",y='time_center',ax=axs1[1],order=["pre_lesion",'1_semaine_postL','3_semaine_postL'],dodge=False, jitter=False,palette=palette,size=7)
    # axs1[1].plot(x,row['time_center'].values,markersize=8,marker="x",c='red')
    # axs1[1].set_ylabel("Temps en pourcentage")
    # axs1[1].set_xlabel(None)

def plot_indiv_bord(rat):
    data=rat_stat(rat,num=6)
    fig1, axs1 = plt.subplots(2,figsize=(8, 10))
    print(data)
    sns.set_theme()

    axs1[0].set_title("Pourcentage de temps passé au bord")
    sns.lineplot(data=data,x="exp",y='time_bord',ax=axs1[0])#,hue='ele_time',,order=["pre_lesion",'1_semaine_postL','3_semaine_postL']
    axs1[0].set_ylabel("Temps en pourcentage")
    axs1[0].set_xlabel(None)

    axs1[1].set_title("Pourcentage de temps passé au centre")
    sns.lineplot(data=data,x="exp",y='time_center',ax=axs1[1])#,hue='ele_time',,order=["pre_lesion",'1_semaine_postL','3_semaine_postL']
    axs1[1].set_ylabel("Temps en pourcentage")
    axs1[1].set_xlabel(None)
    
def plot_rat_indiv(num=6):
    for rat in os.listdir(OUTPUT_PATH)[1:]:
        print(rat)
        plot_rat_stat(rat_stat(rat))
    plt.show()

def paw_count(dataframe):
    yes_left=0
    yes_right=0
    no_left=0
    no_right=0
    L1=dataframe['label_L1'].values
    L2=dataframe['label_L2'].values
    L3=dataframe['label_L3'].values
    L4=dataframe['label_L4'].values
    R1=dataframe['label_R1'].values
    R2=dataframe['label_R2'].values
    R3=dataframe['label_R3'].values
    R4=dataframe['label_R4'].values
    condL=np.array(     [ [float(j) if j not in ['None',' None'] else 0 for j in np.array(i[1:-1].split(","),dtype=object)] for i in dataframe['cam_left_confidence'].values] ,dtype=object)
    condR=np.array(     [ [float(j) if j not in ['None',' None'] else 0 for j in np.array(i[1:-1].split(","),dtype=object)] for i in dataframe['cam_right_confidence'].values] ,dtype=object)
    for i in range(len(L1)):
        labL=0
        labR=0
        L=[L1[i],L2[i],L3[i],L4[i]]
        R=[R1[i],R2[i],R3[i],R4[i]]

        for x in range(4):
            if L[x]==0:
                labL-=condL[i][x]
            elif L[x]==1:
                labL+=condL[i][x]
            if R[x]==0:
                labR-=condR[i][x]
            elif R[x]==1:
                labR+=condR[i][x]
        if labL>0:
            yes_left+=1
        elif labL<0 :
            no_left+=1

        if labR>0:
            yes_right+=1
        elif labR<0 :
            no_right+=1
    return yes_left,no_left,yes_right,no_right

def paw(dataframe):

    L1=dataframe['label_L1'].values
    L2=dataframe['label_L2'].values
    L3=dataframe['label_L3'].values
    L4=dataframe['label_L4'].values
    R1=dataframe['label_R1'].values
    R2=dataframe['label_R2'].values
    R3=dataframe['label_R3'].values
    R4=dataframe['label_R4'].values

    condL=np.array(     [ [float(j) if j not in ['None',' None'] else 0 for j in np.array(i[1:-1].split(","),dtype=object)] for i in dataframe['cam_left_confidence'].values] ,dtype=object)
    condR=np.array(     [ [float(j) if j not in ['None',' None'] else 0 for j in np.array(i[1:-1].split(","),dtype=object)] for i in dataframe['cam_right_confidence'].values] ,dtype=object)

    outL=[]
    outR=[]
    confL=[]
    confR=[]
    for i in range(len(L1)):
        labL=0
        labR=0
        L=[L1[i],L2[i],L3[i],L4[i]]
        R=[R1[i],R2[i],R3[i],R4[i]]

        for x in range(4):
            if L[x]==0:
                labL-=condL[i][x]
            elif L[x]==1:
                labL+=condL[i][x]
            if R[x]==0:
                labR-=condR[i][x]
            elif R[x]==1:
                labR+=condR[i][x]
        confL.append(abs(labL))
        confR.append(abs(labR))
        if labL>0:
            outL.append(1)
        
        elif labL<0 :
            outL.append(0)
        else :
            outL.append(-1)
        if labR>0:
            outR.append(1)
        elif labR<0 :
            outR.append(0)
        else : 
            outR.append(-1)
    return outL,outR,np.mean(confL),np.mean(confR)

def paw_appui_dico(rat,exp,side_lesion,num=0,K=8)  :  
    data=get_csv_paw(rat,exp,num=num)
    L,R=paw_count_ele(data)
    N=len(L)
    OUT={'rat':rat,'exp':exp,'N':N,"double":0,"only_L":0,'only_R':0,'NO':0,'L_r':0,'R_l':0,'L':0,'R':0,"only_lese":0,'only_ok':0,'lese_ok':0,'ok_lese':0,'lese':0,'ok':0,'score':0}
    for deb,fin in zip(L,R):
        #print(data['num'].values[deb])

        frameL,frameR,confL,confR=paw_gauche_droite(data,int(deb),int(fin),K=K)
        #print(confL,confR)
        id=np.argmin([frameL,frameR])
        if side_lesion==['L','R'][id]:
            id_lesion=0
        else :
            id_lesion=1
        if np.isfinite(frameL +frameR):
            if np.abs(frameL-frameR)>10:
                #print(['L_r','R_l'][np.argmin([frameL,frameR])])
                OUT[['L_r','R_l'][id]]+=1
                OUT[['L','R'][id]]+=1
                OUT[['lese_ok','ok_lese'][id_lesion]]+=1
                OUT[['lese','ok'][id_lesion]]+=1

            else :
                OUT['double']+=1
                #print('double')
        elif np.isfinite(min(frameL ,frameR)):

            OUT[['only_L','only_R'][id]]+=1
            OUT[['only_lese','only_ok'][id_lesion]]+=1

            OUT[['L','R'][id]]+=1
            OUT[['lese','ok'][id_lesion]]+=1
           #print(['only_L','only_R'][np.argmin([frameL,frameR])])
        else :

            OUT['NO']+=1
            #print('no')
        #print( ' ' )
    #,"","",'','','','','','',"",'','','','',''
    
    OUT['double_norm']=OUT['double']/N
    OUT['only_L_norm']=OUT['only_L']/N
    OUT['only_R_norm']=OUT['only_R']/N
    OUT['NO_norm']=OUT['NO']/N
    OUT['L_r_norm']=OUT['L_r']/N
    OUT['R_l_norm']=OUT['R_l']/N
    OUT['L_norm']=OUT['L']/N
    OUT['R_norm']=OUT['R']/N
    OUT['only_lese_norm']=OUT['only_lese']/N
    OUT['only_ok_norm']=OUT['only_ok']/N
    OUT['lese_ok_norm']=OUT['lese_ok']/N
    OUT['ok_lese_norm']=OUT['ok_lese']/N
    OUT['lese_norm']=OUT['lese']/N
    OUT['ok_norm']=OUT['ok']/N 
    OUT['score_ninaaaaaa']=((OUT['lese']-OUT['ok'])/(OUT['lese']+OUT['ok']))*100
    OUT['score_ninaaaaaa_only']=((OUT['only_lese']-OUT['only_ok'])/(OUT['only_lese']+OUT['only_ok']))*100
    OUT['score']=((OUT['lese']+0.5*OUT['double'] ) /(OUT['lese']+OUT['double']+OUT['ok']))   
    #print(OUT)
    return OUT

def paw_gauche_droite(dataframe,debut_elevation,fin_elevation,K):
    D=debut_elevation-10 if debut_elevation>10 else 0
    F=fin_elevation +10 if fin_elevation<len(dataframe['num'].values) else len(dataframe['num'].values)-1
    #print(D,F,'D,F')
    T=dataframe['num'].values

    outL,outR,confL,confR=paw(dataframe.iloc[D:F, :])
    
    #print(outR)


    #print(len(outL),'len outL')
    #print('R',outR)
    #print('L',outL)
    T=T[D:F]
    L_start=-1
    R_start=-1

    ret=-1
    
    timeL=np.inf
    timeR=np.inf

    cpt_L=0
    cpt_R=0

    labL=outL[0]
    labR=outR[0]

    for i,(L,R) in enumerate(zip(outL,outR)):
        if L==1 and labL!=1:
            L_start=T[i]
            cpt_L+=1
            labL=1

        if L==1 and labL==1:
            cpt_L+=1
            if cpt_L>K and timeL==np.inf:
                timeL=L_start

        if L==0 and labL==1:
            labL=0
            if cpt_L>=K and timeL==np.inf:
                timeL=L_start
                
            cpt_L=0
        if L==0 and labL!=1:
            labL=0
        
        
        if L==-1 and labL!=1:
            labL=-1



        if R==1 and labR!=1:
            R_start=T[i]
            cpt_R+=1
            labR=1

        if R==1 and labR==1:
            cpt_R+=1
            if cpt_R>K and timeR==np.inf:
                timeR=R_start


        if R==0 and labR==1:
            labR=0
            if cpt_R>=K and timeR==np.inf:
                timeR=R_start
                
            cpt_R=0
        if R==0 and labR!=1:
            labR=0
        
        
        if R==-1 and labR!=1:
            labR=-1
        if timeL!=np.inf and timeR!=np.inf :
            #print(timeL,timeR,'break')
            break
            
    #print(timeL,timeR) 
    return timeL,timeR,confL,confR

def paw_count_ele(dataframe):

    label=paw_ele(dataframe)
    #smooth_y=w_e_smoother.whittaker_smooth(y,lmbd=100,d=1)
    #label=np.where(smooth_y>1.8,1,0)

    
    t=dataframe['num'].values

    c=0
    flag=0


    fin=[]
    debut=[]
    cpt=0
    for i in range(1,len(label)):
        if label[i]==flag:
            cpt+=t[i]-t[i-1]
        else :
            if flag==1 or flag=='elevation':
                #lab.append(t[i])
                fin.append(i)
            else :
                #okok.append(t[i])
                debut.append(i)
            flag=label[i]
            cpt=0
    if len(fin)==len(debut)-1:
        fin.append(t[-1])
    return debut,fin

def paw_ele(dataframe, smooth=True):

    lab_conf=np.array(  np.array([np.array(i[1:-1].split(","),dtype=object).astype(float) for i in dataframe['cam_pose_confidence'].values] ,dtype=object) )
    lab=np.array(     [ [float(j) if j not in ['None',' None'] else -1 for j in np.array(i[1:-1].split(","),dtype=object)] for i in dataframe['cam_pose'].values] ,dtype=object)

    lab_return=[]
    #print(len( dataframe['cam_pose_confidence'].values),len(dataframe['cam_pose'].values))
    for i in range(len(lab)):
        o=0

        for j,l in enumerate(lab[i]):
            o+=(l==0) *-1 *lab_conf[i][j] + (l==1)*1**lab_conf[i][j]
        if o>0:
            lab_return.append(o)
        else :
            lab_return.append(o)
    if smooth :

        lab_return=w_e_smoother.whittaker_smooth(np.array(lab_return),lmbd=100,d=1)
        lab_return=np.where(lab_return>1.8,4,0)  
        lab_return=w_e_smoother.whittaker_smooth(lab_return,lmbd=20,d=1)
        lab_return=np.where(lab_return>2,1,0)
    else :
        lab_return=np.array(lab_return)
    return  lab_return

def run_trajectoire():
    cpt1=np.array([0,0,0,0])
    total_tab=pd.DataFrame(columns=('rat','exp','ele_count','ele_time','ele_norm','distance','number_run','speed_run','total_frame','miss_pts','miss_cam','time_bord','time_center'))
    #print(os.listdir(OUTPUT_PATH)[1:],type(os.listdir(OUTPUT_PATH)))

    
    for rat in os.listdir(OUTPUT_PATH):
        for exp in ['pre_lesion','1_semaine_postL','3_semaine_postL']:
            

            if exp in os.listdir(os.path.join(OUTPUT_PATH,rat)):


            
                print(rat,exp)
                SAVE_PATH=os.path.join(OUTPUT_PATH,rat,exp)

                for i,name in enumerate(os.listdir(SAVE_PATH)):
                    #dataframe=get_csv(rat,exp,num=i)
                    #deb=start(dataframe)
                    
                    dataframe=get_csv_of(rat,exp,num=i)
                    cpt,_=miss_pts(dataframe)
                    cpt_cam,_=miss_cam(dataframe)
                    totalF=time_video(dataframe)
                    
                    x,label=smooth_elevation(dataframe,13)
                    x,label2=smooth_elevation(dataframe,17)
                    x,label3=smooth_elevation(dataframe,21)
                    
                    cpt1=cpt1+np.array(cpt_cam)
                    count1,count2,count3=count(x,label),count(x,label2),count(x,label3)
                    time_ele=elevation_time(x,label)

                    time_ele2=elevation_time(x,label2)
                    time_ele3=elevation_time(x,label3)

                    dist=distance.distance(dataframe)
                    t_bord,t_center=time_bord(dataframe['x'].values,dataframe['y'].values,dataframe['num'].values)
                    
                    x_slow,y_slow,x_fast,y_fast,mean_run_spe=turn.get_slow_fast(dataframe['x'].values,dataframe['y'].values,dataframe['num'].values,seuil=35,windows=1)
                    
                    total_tab.loc[len(total_tab)]={'rat':rat,'exp':exp,'ele_count':np.mean([len(count1),len(count2),len(count3)])/(totalF/60**2),'ele_time':100*np.mean([time_ele,time_ele2,time_ele3])/totalF,'distance':60*dist/totalF,'number_run':len(mean_run_spe)/(totalF/60**2),'speed_run':np.mean(mean_run_spe),'total_frame':totalF,'ele_norm':np.mean([np.mean(count1),np.mean(count2),np.mean(count3)])/60,'time_bord':100*t_bord/(totalF/60),'time_center':100*t_center/(totalF/60),'miss_pts':cpt,'miss_cam':cpt_cam}         
    total_tab.to_csv(os.path.join(ANALYSE_PATH,f"trajectory_result{len(os.listdir(ANALYSE_PATH))}.csv"))

def run_appui():
    
    lesion={'P1B':'L','P2B':'R','P1D':'L','P2D':'R','P1N':'L','P2N':'R','P1R':'L','P2R':'L','P1V':'L','P2V':'R'}
    total_tab=pd.DataFrame(columns=('rat','exp','N',"double","only_L",'only_R','NO','L_r','R_l','L','R',"double_norm","only_L_norm",'only_R_norm','NO_norm','L_r_norm','R_l_norm','L_norm','R_norm',"only_lese",'only_ok','lese_ok','ok_lese','lese','ok',"only_lese_norm",'only_ok_norm','lese_ok_norm','ok_lese_norm','lese_norm','ok_norm','score_ninaaaaaa','score_ninaaaaaa_only'))
    #print(os.listdir(OUTPUT_PATH)[1:],type(os.listdir(OUTPUT_PATH)))

    print(os.listdir(OUTPUT_PATH_PAW))
    for rat in os.listdir(OUTPUT_PATH_PAW):
        for exp in ['pre_lesion','1_semaine_postL','3_semaine_postL']:
            

            if exp in os.listdir(os.path.join(OUTPUT_PATH_PAW,rat)):


            
                print(rat,exp)
                SAVE_PATH=os.path.join(OUTPUT_PATH_PAW,rat,exp)

                for i,name in enumerate(os.listdir(SAVE_PATH)):
                    print(name)
                    dico=paw_appui_dico(rat,exp,lesion[rat],num=i,K=9)
                    data=pd.DataFrame(dico, index=[0])
                    print(dico['N'])
                    total_tab=pd.concat([total_tab.loc[:],data]).reset_index(drop=True)
    total_tab.to_csv(os.path.join(ANALYSE_PATH,f"paw_result{len(os.listdir(ANALYSE_PATH))}.csv")) 
    return total_tab 


def parse_opt():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--source', type=str, help='.MP4 or .avi path file ',required=True)
    parser.add_argument('--mode', type=str, default='test', help='--mode "trajectoire" pour analyser les cvs sorti par chaque rat : , "appui" pour les cvs appui  ',required=True)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    
    with open('./cfg/run_cfg.yaml', 'r') as file :

        dict_cfg = yaml.safe_load(file)

        LESION_SIDE=dict_cfg['SIDE_LESION']

        ANALYSE_PATH=dict_cfg['ANALYSE_PATH']
        if not(os.path.exists(ANALYSE_PATH)):
            print("WARNING : dossier analyse pas trouvé, dossier créé, emplacement : {ANALYSE_PATH}")
            os.makedirs(ANALYSE_PATH)

        OUTPUT_PATH_PAW=dict_cfg['SAVE_PATH_PAW']
        assert os.path.exists(OUTPUT_PATH_PAW), f"dossier résultat appui {OUTPUT_PATH_PAW} pas trouvé"

        OUTPUT_PATH=dict_cfg['SAVE_PATH_TRAJECTORY']
        assert os.path.exists(OUTPUT_PATH), f"dossier résultat trajectoire {OUTPUT_PATH} pas trouvé"
        
        DATA_PATH = dict_cfg['DATA_PATH']

        assert os.path.exists(dict_cfg['DATA_PATH']), f" DATA dir doesn't exist, at {DATA_PATH} !! check documentary for set up the projet's hierachy"

        DATA_CONFIG = dict_cfg['DATA_CONFIG']
        assert os.path.exists(dict_cfg['DATA_CONFIG']) ,f" DATA configuration doesn't exist at {DATA_CONFIG} !! check documentary for set up the projet's hierachy"

        FRAME_RATE=dict_cfg['FRAME_RATE']

        MODEL_PATH=dict_cfg['MODEL_PATH']
        assert os.path.exists(dict_cfg['MODEL_PATH']) ,f" MODEL weights path doesn't exist at {MODEL_PATH} !! check documentary for set up the projet's hierachy"

        #print(set(LESION_SIDE.keys()),set(os.listdir(OUTPUT_PATH_PAW)))
        assert set(LESION_SIDE.keys())==set(os.listdir(OUTPUT_PATH_PAW)) ,f"Les coté de lésion ne corespondent pas au rat précent dans le dossier {OUTPUT_PATH_PAW}"
    if opt.mode =='trajectoire':
        run_trajectoire()
    elif opt.mode=='appui':
        run_appui()
