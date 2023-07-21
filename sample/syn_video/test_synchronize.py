import synchronize
import os 


source1=os.path.join( "data","P1V","pre_lesion","GOPRO 1","GH010112.MP4")
source2=os.path.join( "data","P1V","pre_lesion","GOPRO 2","GH010256.MP4")
source3=os.path.join( "data","P1V","pre_lesion","GOPRO 3","GH010025.MP4")
source4=os.path.join( "data","P1V","pre_lesion","GOPRO 4","GH010037.MP4")

#audio2=os.path.join( "data","P1N","pre_lesion","GOPRO 2","GH010262.mp3")
sources=[source1,source2,source3,source4]
#synchronize.delay_run_4(sources,[600/60,578/60,600/60,460/60],framestart=0)
print(os.path.exists(source2))

#audio2=
print( "...... fps.....;")
fps=int(synchronize.get_fps(source1))
sources=[source1,source2,source3,source4]

print(". . . . . Load Audio 1 . . . . .")
audio1=synchronize.get_audio(source1)
      
print(". . . . . Load Audio 2 . . . . .")
audio2=synchronize.load_audio(source2)
print(". . . . . Load Audio 3 . . . . .")
audio3=synchronize.get_audio(source3)
print(". . . . . Load Audio 4 . . . . .")
audio4=synchronize.get_audio(source4)

audios=[audio1,audio2,audio3,audio4]
print(". . . . . . Cumpute total Delay . . . . . .")

#d=[821.0/60, 1025.0 /60,951.0/60 ,652.0/60]
total_delay=synchronize.global_delay2(audios,fps)
print(total_delay)


synchronize.run_4(sources,total_delay,framestart=5800)