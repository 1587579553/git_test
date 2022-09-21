import numpy as np
A=np.array([[56.0,0.0,4.4,68.0],
           [1.2,104.0,52.0,8.0],
           [1.8,135.0,99.0,0.9]])
cal =A.sum(axis=0)#A.sum（axis=0）表示对矩阵A的每一列进行求和，同理，A.sum（axis=1）表示对A的每一行进行求和
print(cal)
percentage=100*A/cal.reshape(1,4)#reshape命令能让你的矩阵是你想要的形状
print(percentage)