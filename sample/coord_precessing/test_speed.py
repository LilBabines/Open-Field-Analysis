import speed
import os


source=os.path.join("data","P2B","pre_lesion","coordinates","coord.csv")

speed.miss_pts(source)
speed.plot_speed(source)