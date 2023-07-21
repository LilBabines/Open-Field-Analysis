import turn
import speed
import os
import numpy as np
import matplotlib.pyplot as plt

source=os.path.join('E:','Stage_Tremplin','TRAJECTORy','resultat',"P1R",'pre_lesion','coordinates0','coord.csv')

dataframe=turn.load_cvs(source)

x=np.array(dataframe['x'].values)#[ 295 : 401]
y=np.array(dataframe['y'].values)#[295 : 401]
t=np.array(dataframe['num'].values)



x_slow,y_slow,x_fast,y_fast,mean_run_spe=turn.get_slow_fast(x,y,t,l=100,d=1,seuil=70,windows=1)

total_curve=0
curves=[]
#plt.scatter(x_slow,y_slow,c='blue',alpha=0.2,s=1)

for i in range(len(x_fast)):
    a=turn.run_curve(x_fast[i],y_fast[i])
    total_curve+=a
    curves.append(a)
    plt.figure(figsize=(15,15))
    plt.scatter(x,y,c='lightblue',alpha=0.5,s=5)
    plt.scatter(x_fast[i],y_fast[i],c='red',s=5)
    plt.show()


print("total courbe :" ,total_curve)
print(np.argmax(curves))
plt.hist(curves,bins=100)
plt.show()

num=np.argmax(curves)
print()
print(turn.run_curve(x_fast[num],y_fast[num]))
plt.scatter(x,y,c='lightblue',alpha=0.1,s=2)
plt.scatter(x_fast[num],y_fast[num],c='red',s=5)

plt.show()

num=np.argmin(curves)
print()
print(turn.run_curve(x_fast[num],y_fast[num]))
plt.scatter(x,y,c='lightblue',alpha=0.1,s=2)
plt.scatter(x_fast[num],y_fast[num],c='red',s=5)

plt.show()