import distance
import os


source=os.path.join("data","P2B","pre_lesion","coordinates","coord.csv")

dataframe=distance.load_cvs(source)

distance.distance(source)