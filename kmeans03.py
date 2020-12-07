import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  
from math import log ,sqrt,pow
import math
import random
 
 
"""
数据预处理
"""
datanumber=1000 #图像个数
print('读入图片')
def unpickle(file):
    import pickle
    with open(file,'rb')as fo:
        dict =pickle.load(fo,encoding='bytes')
    return dict
#path='../cifar-10-batches-py/'#路径可以自己定义
path="E://机器学习//k-means//cifar-10-python//cifar-10-batches-py//"
a=unpickle(path+'data_batch_1')
temp=np.zeros((10000,3072))
data_label=np.zeros(50000)
data_array=np.zeros((50000,3072))
for i in range(5):
    cur_dict=unpickle(path+'data_batch_'+str(i+1))
    for j in range(10000):
        data_array[i*10000+j]=cur_dict[b'data'][j][:]
        data_label[i*10000+j]=cur_dict[b'labels'][j]

#data_array=data_array.reshape(50000,3,32,32).transpose(0,2,3,1).astype('float32')
data_label=data_label.astype('float32')
 
    
"""
phash图像特征
"""
def pHash(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#创建二维列表
    h,w=gray.shape[:2]
    vis0=np.zeros((h,w),np.float32)
    vis0[:h,:w]=gray
    
#二维DCT变换
    vis1=cv2.dct(cv2.dct(vis0))
    img_list=vis1.flatten()
 
# 计算均值
    avg=sum(img_list)*1./len(img_list)
    avg_list=['0' if i <avg else '1' for i in img_list]
 
#得到哈希值
    hash=''.join(['%x '% int(''.join(avg_list[x:x+4]),2)for x in range(0,32*32,4)])
    hash1=[int(x,16) for x in hash.split()]
    return hash1
 
 
"""
分类图像phash特征提取
"""
true_labels=[]#cifar10真实标签
print('提取phash特征')
feature=[]
for i in range(datanumber):
    #img=data_array[i,:,:,:].astype('uint8')
    true_labels.append(data_label[i].astype('uint8'))
    #feature.append(pHash(img))
print('特征提取结束')
 
 
 
"""
Kmeans分类
"""
#计算欧几里得距离
def distance(vecA,vecB):##符合西瓜书的公式
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
            centnoids[cent]=centtemp.tolist()
        print("kmeans 第%d次 迭代"%time)
        time+=1
    print("kmeans结束")
    predict_labels=clustterassment[:,0].astype('uint8')
    return predict_labels
 
 
"""
性能评估 熵
"""
#n_clusters=10  
#cls=KMeans(n_clusters).fit(feature)
#predict_labels=cls.labels_
predict_labels=kmeans(data_array,10)
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
    norm_c = nc#
    s = 0
    for t in x[i]:
        s += vectorDistance(t,clusters)
    return s/norm_c
 
def compute_Rij(i, j, x, clusters, nc):
    Mij = vectorDistance(clusters[i],clusters[j])
    Rij = (compute_Si(i,x,clusters[i],nc) + compute_Si(j,x,clusters[j],nc))/Mij
    return Rij
 
def compute_Di(i, x, clusters, nc):
    list_r = []
    for j in range(nc):
        if i != j:
            temp = compute_Rij(i, j, x, clusters, nc)
            list_r.append(temp)
    return max(list_r)
 
def compute_DB_index(x, clusters, nc):
    sigma_R = 0.0
    for i in range(nc):
        sigma_R = sigma_R + compute_Di(i, x, clusters, nc)
    DB_index = float(sigma_R)/float(nc)
    return DB_index

DBI2=compute_DB_index(data_array,predict_labels,10)
print('评估矩阵完成,熵为%.5f'%DBI2)
"""
图像聚类可视化
"""
#256维特征feature 通过PCA降维 方便画图
from sklearn.decomposition import PCA  
pca = PCA(n_components=2)             #输出两维  
newData = pca.fit_transform(feature)   #载入N维  
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
