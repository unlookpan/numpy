# numpy第三次打卡
## 根据 sepallength 列对数据集进行排序

```python
from sklearn.datasets import load_iris   #导入数据集iris
iris = load_iris()  #载入数据集
print (iris.data)#打印数据集

'''数据集得排序'''
iris_a=np.array(iris.data)
iris_s = np.lexsort([iris_a[:, 3]])
print('zhishuhou ',iris_a[iris_s])
```
## 在鸢尾属植物数据集中找到最常见的花瓣长度值（第3列）
```python
vals, counts = np.unique(iris_a[:, 2], return_counts=True)
print(vals[np.argmax(counts)])  
print(np.amax(counts))  
```
## 在鸢尾花数据集的 petalwidth（第4列）中查找第一次出现的值大于1.0的位置
```python
#将插入的数字设成1，就能用searchsorted()函数来完成
sarch=np.searchsorted(iris_a[:,3],1)#仅限整数
print(sarch)
```
## 将数组a中大于30的值替换为30，小于10的值替换为10
```python
b = np.where(a < 10, 10, a)
b = np.where(b > 30, 30, b)
print(b)
```
## 获取给定数组a中前5个最大值的位置
```python
a = np.argsort(x)
print(a[0:4])
```
## 计算给定数组中每行的最大值
```python
#取负数，查找最小值
a = np.sort(-1*x,axis=1)
print(-1*a[:,0])
```
## 如何在numpy数组中找到重复值？
```python
b = np.full(10, True)
vals, counts = np.unique(a, return_index=True)
b[counts] = False
print(b)
```
## 删除一维numpy数组中所有NaN值
```python
a = np.array([1, 2, 3, np.nan, 5, 6, 7, np.nan])
b = np.isnan(a)
c = np.where(np.logical_not(b))
print(a[c])
```
## 计算两个数组a和数组b之间的欧氏距离
```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7, 8])

d = np.sqrt(np.sum((a - b) ** 2))
print(d)  # 6.708203932499369
```
## 如何在一维数组中找到所有的局部极大值（或峰值）？
```python
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
b1 = np.diff(a)
b2 = np.sign(b1)
b3 = np.diff(b2)
index = np.where(np.equal(b3, -2))[0] + 1
print(index) # [2 5]
```
## 将numpy的datetime64对象转换为datetime的datetime对象
```python
dt2 = dt64.astype(datetime.datetime)
print(dt2, type(dt2))
```

## 对于给定的一维数组，计算窗口大小为3的移动平均值
```python
a=np.cumsum(b)
for i in range(0:x):
	c[i]=a[((i+2)-(i-1))/3]
print(c)
```
## 创建长度为10的numpy数组，从5开始，在连续的数字之间的步长为3
```python
#第一个参数为起始点
#第二个参数为末点
#第三个参数为步长
a = np.arange(5, 35 , 3)
```
## 将本地图像导入并将其转换为numpy数组
```python
#导入库
from PIL import Image
img1 = Image.open('test.jpg')#打开文件
a = np.array(img1)#转化数组
```





