import paw_yolo
import os

import sys

sys.path.insert(1,"sample")
from syn_video import synchronize




video2=os.path.join('data','P1V','3_semaine_postL','GOPRO 2','GH010182.MP4')
video1=os.path.join('data','P1V','3_semaine_postL','GOPRO 1','GH010182.MP4')
video3=os.path.join('data','P1V','3_semaine_postL','GOPRO 3','GH010056.MP4')
video4=os.path.join('data','P1V','3_semaine_postL','GOPRO 4','GH010060.MP4')
print(os.path.exists(video1))
print(os.path.exists(video2))
print(os.path.exists(video3))
print(os.path.exists(video4))
sources=[video1,video2,video3,video4]
#audio2=os.path.join('data','P2N','pre_lesion','GOPRO 2','GH010261.mp3')

delays=synchronize.global_delay2([synchronize.get_audio(video1),synchronize.get_audio(video2),synchronize.get_audio(video3),synchronize.get_audio(video4)],fps=synchronize.get_fps(video4))
#delays= [578.0//60, 543.0//60, 474.0//60 ,174.0//60]
#synchronize.run_4([video1,video2,video3,video4],delays,framestart=6000)
#Paw_detctoin=paw_yolo.load_model(mode='paw')
#Track_model=paw_yolo.load_model(mode='pose')
#paw_yolo.run2([video1,video2,video3,video4],delays,os.path.join('E:','Stage_Tremplin','PAW','extract_frame'),Paw_detctoin,Track_model,1500,frame_rate_analysis=10)

paw_yolo.test_ele_4(delays,sources,3000)

