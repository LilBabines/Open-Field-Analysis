import mean_smooth
import w_e_smoother
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
source=os.path.join("data","P1D","1_semaine_postL","coordinates","coord.csv")

dataframe=pd.read_csv(source)
x=np.array(dataframe['x'].values)
y=np.array(dataframe['y'].values)


plt.subplot(5,1,1)

plt.title("x coord vs num frame, No smooth")

plt.plot(x)


plt.subplot(5,1,2)

plt.title("--------------------, Mean Smooth, window 5")

plt.plot(mean_smooth.mean_smooth(x,window=5))


plt.subplot(5,1,3)

plt.title("--------------------, Mean Smooth, window 20")

plt.plot(mean_smooth.mean_smooth(x,window=20))
plt.subplot(5,1,4)

plt.title("--------------------, Mean Smooth, window 3")

plt.plot(mean_smooth.mean_smooth(x,window=3))

plt.subplot(5,1,5)

plt.title("x coord vs num frame, w_e smooth, lamda=100")

plt.plot(w_e_smoother.whittaker_smooth( x,100))

plt.show()