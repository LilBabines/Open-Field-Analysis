import features
import os
import numpy as np



save_coord=os.path.join( "data","P2B","pre_lesion","coordinates")
source1=os.path.join( "data","P2B","pre_lesion","GOPRO 1","GH010111.MP4")
source2=os.path.join( "data","P2B","pre_lesion","GOPRO 2","GH010255.MP4")
source3=os.path.join( "data","P2B","pre_lesion","GOPRO 3","GH010024.MP4")
source4=os.path.join( "data","P2B","pre_lesion","GOPRO 4","GH010036.MP4")
sources=[source1,source2,source3,source4]

#audio1=os.path.join( "data","P1B","48h_postL","GOPRO 1","OFGH010144.mp3")
#audio2=os.path.join( "data","P1R","pre_lesion","GOPRO 2","GH010264.mp3")
# audio3=os.path.join( "data","EXP 1","GOPRO 3","audio3.wav")
# audio4=os.path.join( "data","EXP 1","GOPRO 4","audio4.wav")

# audio_source=[audio1,audio2,audio3,audio4]


# H1=np.array([[ 4.62695620e+00 , 1.42329828e+01, -8.32261907e+03],
#  [-4.80702673e+00 , 1.48157768e+01 , 7.53955499e+02],
#  [-1.09602672e-04 , 1.04720054e-02 , 1.00000000e+00]])

# H2=np.array([[ 1.19690467e+00 ,-1.23745119e+00 , 2.77962044e+02],
#  [ 1.24024967e+00 , 3.15484227e+00 ,-1.63330689e+03],
#  [-1.94317813e-07 , 2.01855315e-03 , 1.00000000e+00]])

# H3=np.array([[-3.71568724e+01, -2.72252527e+01  ,7.36551330e+04],
#  [ 3.65617229e+01 ,-3.30313397e+01  ,1.90389698e+03],
#  [ 3.35161225e-03 , 8.31412347e-02 , 1.00000000e+00]])

# H4=np.array([[-4.03026605e+00  ,1.26713532e+01 , 8.41329579e+02],
#  [-4.11543572e+00, -2.97962621e+00 , 7.78356267e+03],
#  [-1.01221319e-04 , 9.25067082e-03 , 1.00000000e+00]])



tracker=features.Rat_tracker(sources,dir_coord_save=save_coord,title="P2B Pre Lesion")#,homographies=[H1,H2,H3,H4,audio1=audio1,audio2=audio2

tracker.test_delay()

tracker.load(5*60+20,frame_start=800,size_tronc=1000)

#tracker.plot()

