import pickle
import numpy as np 
import matplotlib.pyplot as plt
import random
from PIL import Image
from math import log ,sqrt,pow
from sklearn import metrics


import math
path="D://机器学习//cifar-10-batches-py//"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data1=unpickle(path+'data_batch_1')
data1.keys()
print(data1[b"data"].shape)
rand_index=random.randint(0,999)


real_num=500


kdata=data1[b'data']
kdata=kdata[:real_num]
data_label=data1[b'labels']



img=data1[b'data'][rand_index].reshape(3,32,32)
img=img.transpose(1,2,0)
plt.rcParams['figure.dpi']=20
plt.rcParams['figure.figsize']=(4.0,4.0)
plt.imshow(img)




"""
Kmeans分类
"""
#计算欧几里得距离
def distance(vecA,vecB):##符合西瓜书的公式
    return np.sqrt(np.sum(np.square(vecA-vecB)))

def Vdistance(vecA,vecB):##符合西瓜书的公式
    return np.sqrt(np.sum(np.square(vecA-vecB)))
 
#取K个随机质心
def randcent(data,k):##符合西瓜书的公式
    return random.sample(list(data),k)
 
#KMeans实现
def kmeans(dataset,k):
    m=len(dataset)
    clustterassment=np.zeros((m,2)).astype('float32')#二维向量，第一维存放类别 第二维存放距离本类别质心距离
    centnoids=randcent(dataset,k) #随机取K个质心
    clusterchanged=True #用来判断收敛，质心不再变化时，算法停止
    time=1
    while clusterchanged:
        clusterchanged=False
        for i in range(m):
            minDist=float("inf")        
            minIndex=-1
            for j in range(k):
                vec1=np.array(centnoids[j])
                vec2=np.array(dataset[i])
                distji=distance(vec1,vec2)
                if distji<minDist:
                    minDist=distji
                    minIndex=j
            if clustterassment[i,0]!=minIndex:
                clusterchanged=True
            clustterassment[i,:]=minIndex,minDist**2
#更新K个质心        
        for cent in range(k):
            pointsincluster=[]
            for num in range(m):
                if clustterassment[num][0]==cent:
                    pointsincluster.append(dataset[num])#取出同一类别的数据
            centtemp=np.mean(np.array(pointsincluster),axis=0)#计算均值
            centnoids[cent]=centtemp
        print("kmeans 第%d次 迭代"%time)
        time+=1
    print("kmeans结束")
    predict_labels=clustterassment[:,0].astype('uint8')
    return predict_labels,centnoids

true_labels=data_label






predict_labels,centpoint=kmeans(kdata,10)



datanumber=100


entropy=np.zeros(10).astype('float32')
Eall=0
cluster=np.zeros((10,10)).astype('float32')#10个真实标签 10个预测标签 组成10*10的矩阵 存放个数
print("分类图像总数 %d"%datanumber)
for i in range(datanumber):
    cluster[predict_labels[i],true_labels[i]]+=1



for i in range(10):
    Esum=sum(cluster[i,:])#每一类图像总数
    print (cluster[i,:])   
    print("第%d类共%d"%(i,Esum))
         #计算熵    
    for j in range(10):
        p1=cluster[i][j]/Esum 
        if p1!=0:
            entropy[i]+=-(p1*log(p1))
    Eall+=Esum*entropy[i]
Eall=Eall/datanumber
print('评估矩阵完成,熵为%.5f'%Eall)   
 
DBI=metrics.davies_bouldin_score(kdata,predict_labels)

print('评估矩阵完成,DBI为%.5f'%DBI)
 

"""
DBI 自定义函数
"""


def vectorDistance(v1, v2):
    """
    this function calculates de euclidean distance between two
    vectors.
    """
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i]) ** 2
    return sum ** 0.5
 
def compute_Si(i, x, clusters,nc):
    norm_c = nc
    s = 0
    for t in x[i]:
        s += distance(t,clusters)
    return s/norm_c


def compute_avg(i, x, lable_pre):
    s = 0
    index = np.argwhere(np.array(lable_pre) == i)
    n = len(index)
    for i in range(n):
        for j in range(n):
            if i != j:
                s += Vdistance(x[index[i]],x[index[j]])
    return s*2 / (n * (n-1))

def compute_Rij(i, j, x, clusters, nc,cpoint):

    Mij = Vdistance(cpoint[i],cpoint[j])##decn----不同族中心点间的距离
            #  i和j是样本的类别
    Rij = (compute_avg(i,x,clusters) + compute_avg(j,x,clusters))/Mij
    return Rij
 
def compute_Di(i, x, clusters, nc,cpoint):
    list_r = []
    for j in range(nc):
        if i != j:
            temp = compute_Rij(i, j, x, clusters, nc,cpoint)##i和j都是类别
            list_r.append(temp)
    return max(list_r)
 
def compute_DB_index(x, clusters, nc,cpoint):
    sigma_R = 0.0
    for i in range(nc):
        sigma_R = sigma_R + compute_Di(i, x, clusters, nc,cpoint)##i是类别
    DB_index = float(sigma_R)/float(nc)
    return DB_index

DBI2=compute_DB_index(kdata,predict_labels,10,centpoint)

print('评估矩阵完成,自定义DBI为%.5f'%DBI2)

"""
图像聚类可视化
"""
#256维特征feature 通过PCA降维 方便画图
from sklearn.decomposition import PCA  
pca = PCA(n_components=2)             #输出两维  
newData = pca.fit_transform(predict_labels)   #载入N维  
x1=[]
y1=[]
x2=[]
y2=[]
x3=[]
y3=[]
x4=[]
y4=[]
x5=[]
y5=[]
x6=[]
y6=[]
x7=[]
y7=[]
x8=[]
y8=[]
x9=[]
y9=[]
x10=[]
y10=[]
for i in range(datanumber):
    if predict_labels[i]==0:
        x10.append(newData[i][0])
        y10.append(newData[i][1])
    elif predict_labels[i]==1:
        x1.append(newData[i][0])
        y1.append(newData[i][1])
    elif predict_labels[i]==2:
        x2.append(newData[i][0])
        y2.append(newData[i][1])
    elif predict_labels[i]==3:
        x3.append(newData[i][0])
        y3.append(newData[i][1])
    elif predict_labels[i]==4:
        x4.append(newData[i][0])
        y4.append(newData[i][1])
    elif predict_labels[i]==5:
        x5.append(newData[i][0])
        y5.append(newData[i][1])
    elif predict_labels[i]==6:
        x6.append(newData[i][0])
        y6.append(newData[i][1])
    elif predict_labels[i]==7:
        x7.append(newData[i][0])
        y7.append(newData[i][1])
#只取了6个类别给予显示
plt.plot(x1, y1, 'or')  
plt.plot(x2, y2, 'og')   
plt.plot(x3, y3, 'ob')  
plt.plot(x4, y4, 'ok')
plt.plot(x5, y5, 'oy')
plt.plot(x6, y6, 'oc')  
plt.show() 
