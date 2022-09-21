# GRADED FUNCTION: basic_sigmoid

import math
import numpy as np

def basic_sigmoid(x1):#math.exp() 来实现 sigmoid 函数
   ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+(1/math.exp(x1)))

    ### END CODE HERE ###
    print(s)
    return s
basic_sigmoid(3)

def sigmoid(x2):#sigmoid 函数，np.exp()来实现
    s2=1/(1+(1/np.exp(x2)))
    print(s2)

x2=np.array([1,2,3])
sigmoid(x2)


def sigmoid_derivative(x3):#计算Sigmoid 梯度
    s3=1/(1+(1/np.exp(x3)))
    ds=s3*(1-s3)
    print(ds)
    return ds
x3=np.array([1,2,3])
print("sigmoid_derivative(x)="+str(sigmoid_derivative(x3)))

def image2vector(image):#重塑数组
    v=image.reshape((image.shape[0]*image.shape[1]*image.shape[2]),1)
    #print(v)
    return v
image = np.array(
        [[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

        [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

        [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))

def normalizeRows(x4):#归一化是指将 x 更改为X/‖X‖（将 x 的每个行向量除以其范数）。
    x_norm=np.linalg.norm(x4,axis=1,keepdims=True)
    x4=x4/x_norm
    return x4
x4=np.array([[0,3,4],
             [1,6,4]])
print("normalizeRows(x4)"+str(normalizeRows(x4)))

def softmax(x5):#广播和softmax函数
    x_exp=np.exp(x5)
    x_norm=np.sum(x_exp,axis=1,keepdims=True)
    s=x_exp/x_norm
    return s
x5 = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x5) = " + str(softmax(x5)))

def L1(yhat,y):
    loss=abs(y-yhat).sum()
    return  loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

def L2(yhat,y):
    loss1=np.dot(y-yhat,y-yhat)
    return loss1
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))