import pandas as pd
import os



def get_path():
    '''renvoie la config
    emplacement  : ./config.xlsx'''
    path=os.path.join(os.getcwd(),"config.xlsx")
    print(path)
    assert os.path.exists(path)
    print('Load', path)
    return path

def get_video_format(video_name):
    L=video_name.split('.')
    if len(L)==1:
        return video_name+'.MP4'
    else:
        return video_name

def get_video_path(row,dir_data):
    path=[]
    videos=[get_video_format(row['video1']),get_video_format(row['video2']),get_video_format(row['video3']),get_video_format(row['video4'])]
    for i,video in enumerate(videos):
        video_path=os.path.join(dir_data,row['rat'],row['exp'],'GOPRO'+str(i+1),video)
        assert os.path.exists(video_path) 
        path.append(os.path.join(dir_data,row['rat'],row['exp'],'GOPRO'+str(i+1),video))
    return path

def get_df():

    return pd.read_excel(get_path(),sheet_name='config')

def audios(row,dir_data):
    L=[None,None,None,None]
    anomalies=get_anomalie(row)
    
    for i,an in enumerate(anomalies):
        if an =='son':
            L[i]=os.path.join(dir_data,row['rat'],row['exp'],f'GOPRO{i+1}',row[f'video{i+1}'].split('.')[0]+".mp3")
            assert os.path.exists(L[i])
        else :
            L[i]=os.path.join(dir_data,row['rat'],row['exp'],f'GOPRO{i+1}',row[f'video{i+1}']+".MP4")
    return L

def get_anomalie(row):
    return row['anomalies'].split(',')

def get_time(row):
    if row['end_frame']:
        return (row['end_frame'])*60
    else :
        assert row['end_time']
        time=row['end_time']
        return time.minute+time.hour*60

def verif_video(df,dirdata=os.path.join(os.getcwd(),'data')):
    paths=[]
    for index, row in df.iterrows():
        paths.append(get_video_path(row,dirdata))
    return paths

def get_info(row,dir):
    return get_video_path(row,dir),row['exp'],row['rat'],row['rat']+row['exp']