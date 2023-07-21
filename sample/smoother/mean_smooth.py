import numpy as np

def mean_smooth(y,window=5) :
    smooth_y=[np.mean(y[i-(window//2):i+(window//2)+1]) for i in range(window//2,len(y)-(window//2)-1)]
    return smooth_y

