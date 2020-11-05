# numpy第一次打卡
## 创建数组

注意：在线性代数里面讲的维数和数组的维数不同，如线代中提到的n维行向量在 Numpy 中是一维数组，而线性代数中的n维列向量在 Numpy 中是一个shape为(n, 1)的二维数组。

### 数组的属性
ndim:数组的轴数（秩）
shape:行和列（也就是每一维上数据的大小，其返回值是一个元组，元组的长度反应轴数也就是维度）
sise:行乘列
dtype:数组中元素类型
### 创建一维数组
注意：这里的创建一维数组，数组不是python中的数组，而是numpy定义的数组（两者是有区别的，类型不一样）因此要用array（）来转化成numpy中数组的类型。
array()和asarray()都可以将结构数据转化为 ndarray，但是主要区别就是当数据源是 ndarray 时，array()仍然会 copy 出一个副本，占用新的内存，但不改变 dtype 时 asarray()不会。
```python
#创建一个已定数组
a = np.array([0, 1, 2, 3, 4])
#python中自带的数组
s=[1,2,3,4,5]
#转化为numpy中定义的数组
arr = np.arange(s)
print(arr)
```

## 数组的切片与索引
### 副本与视图
副本是从原变量复制出来占用新的内存的变量，而视图则是原变量的引用，占用相同的内存。
numpy中的数组赋值操作与python不同，numpy中的赋值是创建了一个原来变量的引用（视图），而python中，则是创建了一个副本。
```python
numpy.ndarray.copy()#可创建一个副本
```
### 切片
```python
x[start:stop:step]
```
### 索引
索引类似于向量，是在我们定义的数组中，找到其中的值。
```python
x[1,2,3]
#dots 索引

#NumPy 允许使用...表示足够多的冒号来构建完整的索引列表。
x = np.random.randint(1, 100, [2, 2, 3])
print(x)
# [[[ 5 64 75]
#   [57 27 31]]
# 
#  [[68 85  3]
#   [93 26 25]]]

print(x[1, ...])
# [[68 85  3]
#  [93 26 25]]

print(x[..., 2])
# [[75 31]
#  [ 3 25]]

```
## 数组的操作
### 更改形状
shape和reshape（）
```python
#numpy.ndarray.shape略

#numpy.reshape(a, newshape[, order='C'])在不更改数据的情况下为数组赋予新的形状。
x = np.arange(0, 12)
y = np.reshape(x, [3, 4])
print(y, y.dtype)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]] int32
#reshape()函数当参数newshape = -1时表示将数组降为一维。
y = np.reshape(x, -1)
```
转换为一维的迭代器

flat 和 flatten() 和ravel()

```python
'''numpy.ndarray.flat略'''

'''#flatten()返回原数组拷贝，对原数组不做改变'''
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x.flatten()
print(y)
# [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
#  35]

'''ravel()返回的是视图'''
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.ravel(x)
print(y)
# [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
#  35]

```
### 数组转置

T 和 transpose()

```python
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[6.74 8.46 6.74 5.45 1.25]
#  [3.54 3.49 8.62 1.94 9.92]
#  [5.03 7.22 1.6  8.7  0.43]
#  [7.5  7.31 5.69 9.67 7.65]
#  [1.8  9.52 2.78 5.87 4.14]]
y = x.T
print(y)
# [[6.74 3.54 5.03 7.5  1.8 ]
#  [8.46 3.49 7.22 7.31 9.52]
#  [6.74 8.62 1.6  5.69 2.78]
#  [5.45 1.94 8.7  9.67 5.87]
#  [1.25 9.92 0.43 7.65 4.14]]
y = np.transpose(x)
print(y)
# [[6.74 3.54 5.03 7.5  1.8 ]
#  [8.46 3.49 7.22 7.31 9.52]
#  [6.74 8.62 1.6  5.69 2.78]
#  [5.45 1.94 8.7  9.67 5.87]
#  [1.25 9.92 0.43 7.65 4.14]]
```
### 更改维度

```python
'''增加维度numpy.newaxis = None'''
x = np.array([1, 2, 9, 4, 5, 6, 7, 8])
print(x.shape)  # (8,)
print(x)  # [1 2 9 4 5 6 7 8]

y = x[np.newaxis, :]
print(y.shape)  # (1, 8)
print(y)  # [[1 2 9 4 5 6 7 8]]
                                        #区分这两个的区别占时分不行清
y = x[:, np.newaxis]
print(y.shape)  # (8, 1)
print(y)
# [[1]
#  [2]
#  [9]
#  [4]
#  [5]
#  [6]
#  [7]
#  [8]]
'''
numpy.squeeze(a, axis=None) 从数组的形状中删除数据大小为1的条目，即把shape中数据大小为1的维度去掉。
`a`表示输入的数组；
`axis`用于指定需要删除的维度，但是指定的维度的数据大小必须为1，否则将会报错；
单维度：只在此轴上数据等于一
'''
import numpy as np

x = np.array([[[0], [1], [2]]])
print(x.shape)  # (1, 3, 1)
print(x)
# [[[0]
#   [1]
#   [2]]]

y = np.squeeze(x)#表明删除了所有轴上数据为1的维度
print(y.shape)  # (3,)
print(y)  # [0 1 2]

y = np.squeeze(x, axis=0)#删除轴0的数据为1的维度（注意，只能删除数据为1的维度）
print(y.shape)  # (3, 1)
print(y)
# [[0]
#  [1]
#  [2]]

y = np.squeeze(x, axis=2)#删除轴2的数据大小为1的维度
print(y.shape)  # (1, 3)
print(y)  # [[0 1 2]]

y = np.squeeze(x, axis=1)#轴1的大小不是1，在此轴上轴1的数据大小为3
# ValueError: cannot select an axis to squeeze out which has size not equal to one
```

### 组合数组

原维度拼接
```python
'''注意：
原维度拼接的过程，不会改变数组的维度，但是对于多维数组，我们会有按照不同维度的拼接方法
'''
'''一维数组拼接（因为只有一个维度，因此只能按照此维度拼接）'''
x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.concatenate([x, y])
print(z)
# [1 2 3 7 8 9]

'''二维数组的拼接（可以按照轴0和轴1拼接）'''
'''例一'''
x = np.array([1, 2, 3]).reshape(1, 3)
y = np.array([7, 8, 9]).reshape(1, 3)
z = np.concatenate([x, y], axis=0)
print(z)
# [[ 1  2  3]
#  [ 7  8  9]]
z = np.concatenate([x, y], axis=1)
print(z)
# [[ 1  2  3  7  8  9]]
'''例二'''
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
z = np.concatenate([x, y], axis=0)
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
z = np.concatenate([x, y], axis=1)
print(z)
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]
```

增加维度拼接
```python
'''增加维度的拼接，可以增加一个维度，且可以选择对应轴上数据大小的拼接'''
'''一维拼接二维'''
x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.stack([x, y])#默认axis为0
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.stack([x, y], axis=1)
print(z.shape)  # (3, 2)
print(z)
# [[1 7]
#  [2 8]
#  [3 9]]
'''拼接三维'''
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
z = np.stack([x, y])
print(z.shape)  # (2, 2, 3)
print(z)
# [[[ 1  2  3]
#   [ 4  5  6]]
# 
#  [[ 7  8  9]
#   [10 11 12]]]

z = np.stack([x, y], axis=1)
print(z.shape)  # (2, 2, 3)
print(z)
# [[[ 1  2  3]
#   [ 7  8  9]]
# 
#  [[ 4  5  6]
#   [10 11 12]]]

z = np.stack([x, y], axis=2)
print(z.shape)  # (2, 3, 2)
print(z)
# [[[ 1  7]
#   [ 2  8]
#   [ 3  9]]
# 
#  [[ 4 10]
#   [ 5 11]
#   [ 6 12]]]
```

水平方向和垂直方向的拼接
numpy.vstack(tup)垂直方向的拼接
numpy.hstack(tup)水平方向的分割
注意：此组合不分维度，在数据维度等于1时，垂直拼接会增加维度。而当维度大于或等于2时，它们的作用相当于concatenate，用于在已有轴上进行操作。
```python
'''一维'''
import numpy as np

x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.vstack((x, y))
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.stack([x, y])
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.hstack((x, y))
print(z.shape)  # (6,)
print(z)
# [1  2  3  7  8  9]

z = np.concatenate((x, y))
print(z.shape)  # (6,)
print(z)  # [1 2 3 7 8 9]
'''二维'''
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
z = np.vstack((x, y))
print(z.shape)  # (4, 3)
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

z = np.concatenate((x, y), axis=0)
print(z.shape)  # (4, 3)
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

z = np.hstack((x, y))
print(z.shape)  # (2, 6)
print(z)
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]

z = np.concatenate((x, y), axis=1)
print(z.shape)  # (2, 6)
print(z)
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]
```

### 拆分数组
视图拆分与垂直拆分及水平拆分
注意：视图拆分依照维度来拆分的，不同维度的拆分会达到水平拆分和竖直拆分的效果。

视图拆分及水平拆分的比较
```python
x = np.array([[11, 12, 13, 14],
              [16, 17, 18, 19],
              [21, 22, 23, 24]])
y = np.vsplit(x, 3)#垂直拆分
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19]]), array([[21, 22, 23, 24]])]

y = np.split(x, 3)'''默认axis=0'''
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19]]), array([[21, 22, 23, 24]])]


y = np.vsplit(x, [1])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]])]

y = np.split(x, [1])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]])]


y = np.vsplit(x, [1, 3])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]]), array([], shape=(0, 4), dtype=int32)]
y = np.split(x, [1, 3], axis=0)
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]]), array([], shape=(0, 4), dtype=int32)]
```

水平拆分与视图拆分的比较
```python
x = np.array([[11, 12, 13, 14],
              [16, 17, 18, 19],
              [21, 22, 23, 24]])
y = np.hsplit(x, 2)
print(y)
# [array([[11, 12],
#        [16, 17],
#        [21, 22]]), array([[13, 14],
#        [18, 19],
#        [23, 24]])]

y = np.split(x, 2, axis=1)'''将axis设置为1，对轴1进行拆分'''
print(y)
# [array([[11, 12],
#        [16, 17],
#        [21, 22]]), array([[13, 14],
#        [18, 19],
#        [23, 24]])]

y = np.hsplit(x, [3])
print(y)
# [array([[11, 12, 13],
#        [16, 17, 18],
#        [21, 22, 23]]), array([[14],
#        [19],
#        [24]])]

y = np.split(x, [3], axis=1)
print(y)
# [array([[11, 12, 13],
#        [16, 17, 18],
#        [21, 22, 23]]), array([[14],
#        [19],
#        [24]])]

y = np.hsplit(x, [1, 3])
print(y)
# [array([[11],
#        [16],
#        [21]]), array([[12, 13],
#        [17, 18],
#        [22, 23]]), array([[14],
#        [19],
#        [24]])]

y = np.split(x, [1, 3], axis=1)
print(y)
# [array([[11],
#        [16],
#        [21]]), array([[12, 13],
#        [17, 18],
#        [22, 23]]), array([[14],
#        [19],
#        [24]])]
```
### 平铺数组

沿水平和竖直方向复制展开数组

```python
x = np.array([[1, 2], [3, 4]])
print(x)
# [[1 2]
#  [3 4]]

y = np.tile(x, (1, 3))
print(y)
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]]

y = np.tile(x, (3, 1))
print(y)
# [[1 2]
#  [3 4]
#  [1 2]
#  [3 4]
#  [1 2]
#  [3 4]]

y = np.tile(x, (3, 3))
print(y)
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]]
```
灵活的复制数组中的值
numpy.repeat(a, repeats, axis=None) Repeat elements of an array.
`axis=0`，沿着y轴复制，实际上增加了行数。

`axis=1`，沿着x轴复制，实际上增加了列数。

`repeats`，可以为一个数，也可以为一个矩阵。

`axis=None`时就会flatten当前矩阵，实际上就是变成了一个行向量。
```python
x = np.repeat(3, 4)
print(x)  # [3 3 3 3]

x = np.array([[1, 2], [3, 4]])
y = np.repeat(x, 2)
print(y)
# [1 1 2 2 3 3 4 4]

y = np.repeat(x, 2, axis=0)
print(y)
# [[1 2]
#  [1 2]
#  [3 4]
#  [3 4]]

y = np.repeat(x, 2, axis=1)
print(y)
# [[1 1 2 2]
#  [3 3 4 4]]

y = np.repeat(x, [2, 3], axis=0)
print(y)
# [[1 2]
#  [1 2]
#  [3 4]
#  [3 4]
#  [3 4]]

y = np.repeat(x, [2, 3], axis=1)
print(y)
# [[1 1 2 2 2]
#  [3 3 4 4 4]]
```

## 运算函数
### 基本运算符
做一些对数组的加减操作
### 三角函数
返回三角函数值
### 加法与乘法函数
numpy.sum(a[, axis=None, dtype=None, out=None, …])
可以沿着不同的轴向进行求和
```python
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.sum(x)
print(y)  # 575

y = np.sum(x, axis=0)
print(y)  # [105 110 115 120 125]

y = np.sum(x, axis=1)
print(y)  # [ 65  90 115 140 165]
```
可以显示数组沿轴向的累加
numpy.cumsum(a, axis=None, dtype=None, out=None)
```python
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.cumsum(x)
print(y)
# [ 11  23  36  50  65  81  98 116 135 155 176 198 221 245 270 296 323 351
#  380 410 441 473 506 540 575]

y = np.cumsum(x, axis=0)
print(y)
# [[ 11  12  13  14  15]
#  [ 27  29  31  33  35]
#  [ 48  51  54  57  60]
#  [ 74  78  82  86  90]
#  [105 110 115 120 125]]

y = np.cumsum(x, axis=1)
print(y)
# [[ 11  23  36  50  65]
#  [ 16  33  51  70  90]
#  [ 21  43  66  90 115]
#  [ 26  53  81 110 140]
#  [ 31  63  96 130 165]]
```
乘法用法同理：
返回轴向上的乘积：numpy.prod(a[, axis=None, dtype=None, out=None, …])
返回沿某一轴向上的累乘：numpy.cumprod(a, axis=None, dtype=None, out=None)

### 四舍五入
numpy.around(a, decimals=0, out=None)
```python
x = np.random.rand(3, 3) * 10
print(x)
# [[6.59144457 3.78566113 8.15321227]
#  [1.68241475 3.78753332 7.68886328]
#  [2.84255822 9.58106727 7.86678037]]

y = np.around(x)
print(y)
# [[ 7.  4.  8.]
#  [ 2.  4.  8.]
#  [ 3. 10.  8.]]

y = np.around(x, decimals=2)
print(y)
# [[6.59 3.79 8.15]
#  [1.68 3.79 7.69]
#  [2.84 9.58 7.87]]
```


