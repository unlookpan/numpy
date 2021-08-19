'''手写数字识别算法'''
import numpy as np 
from sklearn import datasets 
import matplotlib.pyplot as plt

'''初始化基本量'''
TP=0  #判定为正确地正样本
EX=0 #每个数字比较的方差
suan=[i for i in range(0,1797)] #存储一个数字数据与所有数据比较后的方差数组

'''获取数据集并打印基本参数'''
datas = datasets.load_digits()
#打印数据集数据及目标值的数组大小
print("数据集的大小：",datas.data.shape)
print("数据集目标值的大小：",datas.target.shape)

#单个数据的测试利用代码
#plt.gray() 
#plt.matshow(datas.images[bie]) 
#plt.show() 
#print(datas.target[1])

'''数据计算'''
#分道处理数据是1347
for b in range(0,1797):                  #循环遍历所有数据进行手写数字识别
    print("正在检测得数字得正确答案：",datas.target[b])            #输出正在检测的数字目标值
    #print(datas.data[b])              #输出正在检测的数据
    for i in range(0,1797):           #对一个数组进行对全部数据的比较计算
        for j in range(0,64):         #一次比较对64个像素全部计比较
            EX=EX+((datas.data[b][j]-datas.data[i][j])**2)
        suan[i]=EX
        EX=0  
    #argsort()返回数组中数据排序后的索引
    suans=np.asarray(suan)
    y=np.argsort(suans)    #将得到的数组进行转化并排序返回排序后的索引
    #bincount()返回索引数组中每个元素出现的次数
    #argmax()则返回一个数组中的最大值，此处返回索引出现次数的最大值
    #print(suans[y])
    #array1 = y[0:1700]
    #np.argmax()
    #print('lllllll',y.shape)
    c_1=np.argmax(np.bincount(datas.target[y[0:20]]))    #返回索引对应数据出现的次数，并返回出现次数的最大值
    print("返回出现次数",np.bincount(datas.target[y[0:20]]))
    #索引出现次数最多对应的值则是我们想要的目标值
    if(datas.target[b]==c_1):
        TP=TP+1
    #打印每次检验的最终结果
    print("检测结果为：",c_1)
#print(datas.target[y[0:100]])
print("检测结果对的数目：",TP)


    
