import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time as t

import argparse
import sys 
import yaml
sys.path.insert(1,"sample")
from paw_detection import paw_yolo
from syn_video import synchronize
sys.path.insert(1,"utils")
import read_exel

def run():
    start = t.time()
    #fichier config (nom des video anomalis, durée, etc.)
    df_config=read_exel.get_df(cfg=DATA_CONFIG)
    FRAME_START=800

    #vérifie si tout les video du fichier config sont trouvées
    read_exel.verif_video(df_config)

    #chargement du model, YOLO
    Paw_detction=paw_yolo.load_model(mode='paw')
    Track_model=paw_yolo.load_model(mode='pose')






    for index,row in df_config.iterrows():
        start = t.time()
        print("Début")

        

        sources,exp,rat,title=read_exel.get_info(row,DATA_PATH)

        current_path=os.path.join(SAVE_PATH,rat,exp)

        if not(os.path.exists(current_path)):
            os.makedirs(current_path)

        #création d'un dossier coordinates{i}, pour i tout les calcul d'une même expérience (ou si deux prélésion d'un même rat par exmeple)
        SAVE_DIR=os.path.join(current_path,'coordinates'+str(len(os.listdir(current_path))))
        os.makedirs(SAVE_DIR)



       

        audios=read_exel.audios(row,DATA_PATH)


        anomalis=read_exel.get_anomalie(row)
        son=[1,1,1,1]
        print(rat,exp)


        audio_fps=synchronize.get_fps(sources[3])
        #print(audio_fps)
        for i,ano in enumerate(anomalis):
            #print(i)
            if ano=='son':
                #print(audios[i])
                #print(os.path.exists(audios[i]))
                audios[i]=synchronize.load_audio(audios[i])
                son[i]=0
                #pass
            else :
                #print(os.path.exists(audios[i]))
                audios[i]=synchronize.get_audio(audios[i])
                #pass

        #delays=[1494.0/60, 1403.0/60, 1220.0/60, 1110.0/60]
        delays=synchronize.global_delay2(audios,fps=audio_fps)
        
        
        
        paw_yolo.run(sources,delays,son,SAVE_DIR,Paw_detction,Track_model,frame_start=FRAME_START,frame_rate_analysis=FRAME_RATE)

        
        end = t.time()
        print('end',end - start)
        

def test_yolo():
    df_config=read_exel.get_df(cfg=DATA_CONFIG)
    FRAME_START=800

    #vérifie si tout les video du fichier config sont trouvées
    read_exel.verif_video(df_config)
    row=df_config.sample(1)
    sources,exp,rat,title=read_exel.get_info(row,DATA_PATH)
    paw_yolo.test(sources[3],4000,frame_rate_analysis=4)

def parse_opt():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--source', type=str, help='.MP4 or .avi path file ',required=True)
    parser.add_argument('--mode', type=str, default='test', help='test_yolo pour tester le modèle sur un video aléatoire : , "run" pour calculer toute les expérience présente dans le DATA_CONFIG  ',required=True)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    
    with open('./cfg/run_cfg.yaml', 'r') as file :

        dict_cfg = yaml.safe_load(file)

        SAVE_PATH=dict_cfg['SAVE_PATH_PAW']
        if not(os.path.exists(SAVE_PATH)):
            print(f"WARNING : save path doesn't not exist , new one created at {SAVE_PATH}")
            os.makedirs(SAVE_PATH)

        DATA_PATH = dict_cfg['DATA_PATH']

        assert os.path.exists(dict_cfg['DATA_PATH']), f" DATA dir doesn't exist, at {DATA_PATH} !! check documentary for set up the projet's hierachy"

        DATA_CONFIG = dict_cfg['DATA_CONFIG']
        assert os.path.exists(dict_cfg['DATA_CONFIG']) ,f" DATA configuration doesn't exist at {DATA_CONFIG} !! check documentary for set up the projet's hierachy"

        FRAME_RATE=dict_cfg['FRAME_RATE']

        MODEL_PATH=dict_cfg['MODEL_PATH']
        assert os.path.exists(dict_cfg['MODEL_PATH']) ,f" MODEL weights path doesn't exist at {MODEL_PATH} !! check documentary for set up the projet's hierachy"

    if opt.mode =='test':
        test_yolo()
    elif opt.mode=='run':
        run()
