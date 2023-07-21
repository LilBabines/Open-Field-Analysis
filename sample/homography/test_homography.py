from homography import findHomography,run,save
import os
import numpy as np
source1=os.path.join( "data","P1N","pre_lesion","GOPRO 1","GH010118.MP4")
source2=os.path.join( "data","P1N","pre_lesion","GOPRO 2","GH010262.avi")
source3=os.path.join( "data","P1N","pre_lesion","GOPRO 3","GH010033.MP4")
source4=os.path.join( "data","P1N","pre_lesion","GOPRO 4","GH010043.MP4")
print(os.path.exists(source2))
dir=os.path.join("data","P1N","pre_lesion",'homography')

H1=findHomography(source1)
save(H1,'H1',dir=dir)

H2=findHomography(source2)
save(H2,'H2',dir=dir)

H3=findHomography(source3)
save(H3,'H3',dir=dir)


H4=findHomography(source4)
save(H4,'H4',dir=dir)





#H=np.array([[-4.19603703e+01, -9.41856349e+01 , 6.32254876e+04],[ 2.23215610e+01, -1.03369071e+02 , 5.46365820e+03],[-7.03041835e-03, -7.53806468e-02,  1.00000000e+00]],dtype=np.float32)

