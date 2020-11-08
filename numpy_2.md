# numpy_2
## 逻辑函数
### 真值测试
```python
'''
判断两个数组中相等的值（all，any）返回布尔值
'''
a = np.array([0, 4, 5])
b = np.copy(a)
print(np.all(a == b))  # True
print(np.any(a == b))  # True
b[0] = 1
print(np.all(a == b))  # False（全部相等）
print(np.any(a == b))  # True（存在一个元素相等）

#不解
print(np.all([1.0, np.nan]))  # True
print(np.any([1.0, np.nan]))  # True

a = np.eye(3)
print(np.all(a, axis=0))  # [False False False]
print(np.any(a, axis=0))  # [ True  True  True]
```
### 逻辑运算
该函数的返回值为布尔类型，是用来进行矩阵之间逻辑运算的布尔结果
```python
x = np.arange(5)
'''与，或，非，异或运算'''
print(np.logical_not(3))  
# False
print(np.logical_not([True, False, 0, 1]))
# [False  True  True False]
print(np.logical_not(x < 3))
# [False False False  True  True]

print(np.logical_and(True, False))  
# False
print(np.logical_and([True, False], [True, False]))
# [ True False]
print(np.logical_and(x > 1, x < 4))
# [False False  True  True False]

print(np.logical_or(True, False))
# True
print(np.logical_or([True, False], [False, False]))
# [ True False]
print(np.logical_or(x < 1, x > 3))
# [ True False False False  True]

print(np.logical_xor(True, False))
# True
print(np.logical_xor([True, True, False, False], [True, False, True, False]))
# [False  True  True False]
print(np.logical_xor(x < 1, x > 3))
# [ True False False False  True]
print(np.logical_xor(0, np.eye(2)))
# [[ True False]
#  [False  True]]
```
### 对照
对两个矩阵进行比较返回布尔类型结果（矩阵）
```python
'''维数组与数字的比较'''
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

y = x > 2
print(y)
print(np.greater(x, 2))
# [False False  True  True  True  True  True  True]

y = x >= 2
print(y)
print(np.greater_equal(x, 2))
# [False  True  True  True  True  True  True  True]

y = x == 2
print(y)
print(np.equal(x, 2))
# [False  True False False False False False False]

y = x != 2
print(y)
print(np.not_equal(x, 2))
# [ True False  True  True  True  True  True  True]

y = x < 2
print(y)
print(np.less(x, 2))
# [ True False False False False False False False]

y = x <= 2
print(y)
print(np.less_equal(x, 2))
# [ True  True False False False False False False]

'''二维数组与数字的比较'''
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x > 20
print(y)
print(np.greater(x, 20))
# [[False False False False False]
#  [False False False False False]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]

y = x >= 20
print(y)
print(np.greater_equal(x, 20))
# [[False False False False False]
#  [False False False False  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]

y = x == 20
print(y)
print(np.equal(x, 20))
# [[False False False False False]
#  [False False False False  True]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]]

y = x != 20
print(y)
print(np.not_equal(x, 20))
# [[ True  True  True  True  True]
#  [ True  True  True  True False]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]


y = x < 20
print(y)
print(np.less(x, 20))
# [[ True  True  True  True  True]
#  [ True  True  True  True False]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]]

y = x <= 20
print(y)
print(np.less_equal(x, 20))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]]

'''二维数组间的比较'''
np.random.seed(20200611)
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = np.random.randint(10, 40, [5, 5])
print(y)
# [[32 28 31 33 37]
#  [23 37 37 30 29]
#  [32 24 10 33 15]
#  [27 17 10 36 16]
#  [25 32 23 39 34]]

z = x > y
print(z)
print(np.greater(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False  True False  True]
#  [False  True  True False  True]
#  [ True False  True False  True]]

z = x >= y
print(z)
print(np.greater_equal(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False  True False  True]
#  [False  True  True False  True]
#  [ True  True  True False  True]]

z = x == y
print(z)
print(np.equal(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]
#  [False  True False False False]]

z = x != y
print(z)
print(np.not_equal(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True False  True  True  True]]

z = x < y
print(z)
print(np.less(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True False  True False]
#  [ True False False  True False]
#  [False False False  True False]]

z = x <= y
print(z)
print(np.less_equal(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True False  True False]
#  [ True False False  True False]
#  [False  True False  True False]]

'''二维数组与一维数组的比较'''
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

np.random.seed(20200611)
y = np.random.randint(10, 50, 5)

print(y)
# [32 37 30 24 10]

z = x > y
print(z)
print(np.greater(x, y))
# [[False False False False  True]
#  [False False False False  True]
#  [False False False False  True]
#  [False False False  True  True]
#  [False False  True  True  True]]

z = x >= y
print(z)
print(np.greater_equal(x, y))
# [[False False False False  True]
#  [False False False False  True]
#  [False False False  True  True]
#  [False False False  True  True]
#  [False False  True  True  True]]

z = x == y
print(z)
print(np.equal(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False False  True False]
#  [False False False False False]
#  [False False False False False]]

z = x != y
print(z)
print(np.not_equal(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True False  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]

z = x < y
print(z)
print(np.less(x, y))
# [[ True  True  True  True False]
#  [ True  True  True  True False]
#  [ True  True  True False False]
#  [ True  True  True False False]
#  [ True  True False False False]]

z = x <= y
print(z)
print(np.less_equal(x, y))
# [[ True  True  True  True False]
#  [ True  True  True  True False]
#  [ True  True  True  True False]
#  [ True  True  True False False]
#  [ True  True False False False]]

'''比较两个数组是否能近似相等'''
x = np.isclose([1e10, 1e-7], [1.00001e10, 1e-8])
print(x)  # [ True False]

x = np.allclose([1e10, 1e-7], [1.00001e10, 1e-8])
print(x)  # False

x = np.isclose([1e10, 1e-8], [1.00001e10, 1e-9])
print(x)  # [ True  True]

x = np.allclose([1e10, 1e-8], [1.00001e10, 1e-9])
print(x)  # True

x = np.isclose([1e10, 1e-8], [1.0001e10, 1e-9])
print(x)  # [False  True]

x = np.allclose([1e10, 1e-8], [1.0001e10, 1e-9])
print(x)  # False

x = np.isclose([1.0, np.nan], [1.0, np.nan])
print(x)  # [ True False]

x = np.allclose([1.0, np.nan], [1.0, np.nan])
print(x)  # False

x = np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
print(x)  # [ True  True]

x = np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
print(x)  # True
```
## 排序
```python
'''注意使用：
x:参数为数组
axis:排序沿数组的（轴）方向，0表示按行，1表示按列，None表示展开来排序，默认为-1，表示沿最后的轴排序。
kind:排序的算法，提供了快排'quicksort'、混排'mergesort'、堆排'heapsort'， 默认为‘quicksort'.
order:排序的字段名，可指定字段排序，默认为None
'''
np.random.seed(20200612)
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[2.32 7.54 9.78 1.73 6.22]
#  [6.93 5.17 9.28 9.76 8.25]
#  [0.01 4.23 0.19 1.73 9.27]
#  [7.99 4.97 0.88 7.32 4.29]
#  [9.05 0.07 8.95 7.9  6.99]]

y = np.sort(x)
print(y)
# [[1.73 2.32 6.22 7.54 9.78]
#  [5.17 6.93 8.25 9.28 9.76]
#  [0.01 0.19 1.73 4.23 9.27]
#  [0.88 4.29 4.97 7.32 7.99]
#  [0.07 6.99 7.9  8.95 9.05]]

y = np.sort(x, axis=0)
print(y)
# [[0.01 0.07 0.19 1.73 4.29]
#  [2.32 4.23 0.88 1.73 6.22]
#  [6.93 4.97 8.95 7.32 6.99]
#  [7.99 5.17 9.28 7.9  8.25]
#  [9.05 7.54 9.78 9.76 9.27]]

y = np.sort(x, axis=1)
print(y)
# [[1.73 2.32 6.22 7.54 9.78]
#  [5.17 6.93 8.25 9.28 9.76]
#  [0.01 0.19 1.73 4.23 9.27]
#  [0.88 4.29 4.97 7.32 7.99]
#  [0.07 6.99 7.9  8.95 9.05]]

'''展示字段排序'''
dt = np.dtype([('name', 'S10'), ('age', np.int)])
a = np.array([("Mike", 21), ("Nancy", 25), ("Bob", 17), ("Jane", 27)], dtype=dt)
b = np.sort(a, order='name')
print(b)
# [(b'Bob', 17) (b'Jane', 27) (b'Mike', 21) (b'Nancy', 25)]

b = np.sort(a, order='age')
print(b)
# [(b'Bob', 17) (b'Mike', 21) (b'Nancy', 25) (b'Jane', 27)]
```
排序返回排序后的索引数组，用来构建排序后的数组
```python
'''该排序是返回索引'''
import numpy as np

np.random.seed(20200612)
x = np.random.randint(0, 10, 10)
print(x)
# [6 1 8 5 5 4 1 2 9 1]

y = np.argsort(x)
print(y)
# [1 6 9 7 5 3 4 0 2 8]

print(x[y])
# [1 1 1 2 4 5 5 6 8 9]

y = np.argsort(-x)
print(y)
# [8 2 0 3 4 5 7 1 6 9]

print(x[y])
# [9 8 6 5 5 4 2 1 1 1]

'''二维的排序'''
import numpy as np

np.random.seed(20200612)
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[2.32 7.54 9.78 1.73 6.22]
#  [6.93 5.17 9.28 9.76 8.25]
#  [0.01 4.23 0.19 1.73 9.27]
#  [7.99 4.97 0.88 7.32 4.29]
#  [9.05 0.07 8.95 7.9  6.99]]

y = np.argsort(x)
print(y)
# [[3 0 4 1 2]
#  [1 0 4 2 3]
#  [0 2 3 1 4]
#  [2 4 1 3 0]
#  [1 4 3 2 0]]

y = np.argsort(x, axis=0)
print(y)
# [[2 4 2 0 3]
#  [0 2 3 2 0]
#  [1 3 4 3 4]
#  [3 1 1 4 1]
#  [4 0 0 1 2]]

y = np.argsort(x, axis=1)
print(y)
# [[3 0 4 1 2]
#  [1 0 4 2 3]
#  [0 2 3 1 4]
#  [2 4 1 3 0]
#  [1 4 3 2 0]]

y = np.array([np.take(x[i], np.argsort(x[i])) for i in range(5)])
print(y)
# [[1.73 2.32 6.22 7.54 9.78]
#  [5.17 6.93 8.25 9.28 9.76]
#  [0.01 0.19 1.73 4.23 9.27]
#  [0.88 4.29 4.97 7.32 7.99]
#  [0.07 6.99 7.9  8.95 9.05]]
```
按照某一指标排序（比如按照第一列的大小给整个数组排序）
```python
import numpy as np

np.random.seed(20200612)
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[2.32 7.54 9.78 1.73 6.22]
#  [6.93 5.17 9.28 9.76 8.25]
#  [0.01 4.23 0.19 1.73 9.27]
#  [7.99 4.97 0.88 7.32 4.29]
#  [9.05 0.07 8.95 7.9  6.99]]

y = np.argsort(x)
print(y)
# [[3 0 4 1 2]
#  [1 0 4 2 3]
#  [0 2 3 1 4]
#  [2 4 1 3 0]
#  [1 4 3 2 0]]

y = np.argsort(x, axis=0)
print(y)
# [[2 4 2 0 3]
#  [0 2 3 2 0]
#  [1 3 4 3 4]
#  [3 1 1 4 1]
#  [4 0 0 1 2]]

y = np.argsort(x, axis=1)
print(y)
# [[3 0 4 1 2]
#  [1 0 4 2 3]
#  [0 2 3 1 4]
#  [2 4 1 3 0]
#  [1 4 3 2 0]]

y = np.array([np.take(x[i], np.argsort(x[i])) for i in range(5)])
print(y)
# [[1.73 2.32 6.22 7.54 9.78]
#  [5.17 6.93 8.25 9.28 9.76]
#  [0.01 0.19 1.73 4.23 9.27]
#  [0.88 4.29 4.97 7.32 7.99]
#  [0.07 6.99 7.9  8.95 9.05]]

'''复杂条件排序'''
import numpy as np

x = np.array([1, 5, 1, 4, 3, 4, 4])
y = np.array([9, 4, 0, 4, 0, 2, 1])
a = np.lexsort([x])
b = np.lexsort([y])
print(a)
# [0 2 4 3 5 6 1]
print(x[a])
# [1 1 3 4 4 4 5]

print(b)
# [2 4 6 5 1 3 0]
print(y[b])
# [0 0 1 2 4 4 9]

z = np.lexsort([y, x])
print(z)
# [2 0 4 6 5 3 1]
print(x[z])
# [1 1 3 4 4 4 5]

z = np.lexsort([x, y])
print(z)
# [2 4 6 5 3 1 0]
print(y[z])
# [0 0 1 2 4 4 9]
```
## 搜索
返回最大值
```python
import numpy as np

np.random.seed(20200612)
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[2.32 7.54 9.78 1.73 6.22]
#  [6.93 5.17 9.28 9.76 8.25]
#  [0.01 4.23 0.19 1.73 9.27]
#  [7.99 4.97 0.88 7.32 4.29]
#  [9.05 0.07 8.95 7.9  6.99]]

y = np.argmax(x)
print(y)  # 2

y = np.argmax(x, axis=0)
print(y)
# [4 0 0 1 2]

y = np.argmax(x, axis=1)
print(y)
# [2 3 4 0 0]
```
返回最小值
```python
import numpy as np

np.random.seed(20200612)
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[2.32 7.54 9.78 1.73 6.22]
#  [6.93 5.17 9.28 9.76 8.25]
#  [0.01 4.23 0.19 1.73 9.27]
#  [7.99 4.97 0.88 7.32 4.29]
#  [9.05 0.07 8.95 7.9  6.99]]

y = np.argmin(x)
print(y)  # 10

y = np.argmin(x, axis=0)
print(y)
# [2 4 2 0 3]

y = np.argmin(x, axis=1)
print(y)
# [3 1 0 2 1]
```
返回给定条件的索引
```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.where(x > 5)
print(y)
# (array([5, 6, 7], dtype=int64),)
print(x[y])
# [6 7 8]

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.where(x > 25)
print(y)
# (array([3, 3, 3, 3, 3, 4, 4, 4, 4, 4], dtype=int64), array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int64))

print(x[y])
# [26 27 28 29 30 31 32 33 34 35]
```
若想插入元素且不改变原来的排序顺序且返回其索引
```python
import numpy as np

x = np.array([0, 1, 5, 9, 11, 18, 26, 33])
y = np.searchsorted(x, 15)
print(y)  # 5

y = np.searchsorted(x, 15, side='right')
print(y)  # 5

'''其他参数例程'''
import numpy as np

x = np.array([0, 1, 5, 9, 11, 18, 26, 33])
y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35])
print(y)  # [0 0 4 5 7 8]

y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35], side='right')
print(y)  # [0 1 5 5 8 8]

'''其他'''
import numpy as np

x = np.array([0, 1, 5, 9, 11, 18, 26, 33])
np.random.shuffle(x)
print(x)  # [33  1  9 18 11 26  0  5]

x_sort = np.argsort(x)
print(x_sort)  # [6 1 7 2 4 3 5 0]

y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35], sorter=x_sort)
print(y)  # [0 0 4 5 7 8]

y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35], side='right', sorter=x_sort)
print(y)  # [0 1 5 5 8 8]
```
### 计数
```python
'''返回非零元素的个数'''
import numpy as np

x = np.count_nonzero(np.eye(4))
print(x)  # 4

x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
print(x)  # 5

x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]], axis=0)
print(x)  # [1 1 1 1 1]

x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]], axis=1)
print(x)  # [2 3]
```
## 随机抽样
```python
'''
numpy.random.seed(seed=None) Seed the generator.

seed()用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()值，则每次生成的随机数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。

在对数据进行预处理时，经常加入新的操作或改变处理策略，此时如果伴随着随机操作，最好还是指定唯一的随机种子，避免由于随机的差异对结果产生影响。
'''
```
其余随机函数掠过

## 集合操作
```python
'''找出数组中的唯一值返回已排序的结果'''
import numpy as np

x = np.unique([1, 1, 3, 2, 3, 3])
print(x)  # [1 2 3]

x = sorted(set([1, 1, 3, 2, 3, 3]))
print(x)  # [1, 2, 3]

x = np.array([[1, 1], [2, 3]])
u = np.unique(x)
print(u)  # [1 2 3]

x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
y = np.unique(x, axis=0)
print(y)
# [[1 0 0]
#  [2 3 4]]

x = np.array(['a', 'b', 'b', 'c', 'a'])
u, index = np.unique(x, return_index=True)
print(u)  # ['a' 'b' 'c']
print(index)  # [0 1 3]
print(x[index])  # ['a' 'b' 'c']

x = np.array([1, 2, 6, 4, 2, 3, 2])
u, index = np.unique(x, return_inverse=True)
print(u)  # [1 2 3 4 6]
print(index)  # [0 1 4 3 1 2 1]
print(u[index])  # [1 2 6 4 2 3 2]

u, count = np.unique(x, return_counts=True)
print(u)  # [1 2 3 4 6]
print(count)  # [1 3 1 1 1]
```
### 布尔运算
```python
'''
判断前面的数组是否包含于后面的数组，返回布尔值。返回的值是针对第一个参数的数组的，所以维数和第一个参数一致，布尔值与数组的元素位置也一一对应。
'''
import numpy as np

test = np.array([0, 1, 2, 5, 0])
states = [0, 2]
mask = np.in1d(test, states)
print(mask)  # [ True False  True False  True]
print(test[mask])  # [0 2 0]

mask = np.in1d(test, states, invert=True)
print(mask)  # [False  True False  True False]
print(test[mask])  # [1 5]
```
```python
'''求两个集合的交集'''
import numpy as np
from functools import reduce

x = np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
print(x)  # [1 3]

x = np.array([1, 1, 2, 3, 4])
y = np.array([2, 1, 4, 6])
xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
print(x_ind)  # [0 2 4]
print(y_ind)  # [1 0 2]
print(xy)  # [1 2 4]
print(x[x_ind])  # [1 2 4]
print(y[y_ind])  # [1 2 4]

x = reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
print(x)  # [3]

'''求两个集合的并集'''
import numpy as np
from functools import reduce

x = np.union1d([-1, 0, 1], [-2, 0, 2])
print(x)  # [-2 -1  0  1  2]
x = reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
print(x)  # [1 2 3 4 6]

'''球两个集合的差集'''
import numpy as np

a = np.array([1, 2, 3, 2, 4, 1])
b = np.array([3, 4, 5, 6])
x = np.setdiff1d(a, b)
print(x)  # [1 2]

'''两个集合的异或'''
import numpy as np

a = np.array([1, 2, 3, 2, 4, 1])
b = np.array([3, 4, 5, 6])
x = np.setxor1d(a, b)
print(x)  # [1 2 5 6]
```

## 总结
```python
'''判断量数组中是否有同一变量（结果返回布尔值）'''
numpy.all(a, axis=None, out=None, keepdims=np._NoValue)
numpy.any(a, axis=None, out=None, keepdims=np._NoValue)

'''对两个矩阵进行逻辑运算返回布尔值矩阵'''
numpy.logical_not(x, *args, **kwargs)
numpy.logical_and(x1, x2, *args, **kwargs)
numpy.logical_or(x1, x2, *args, **kwargs)
numpy.logical_xor(x1, x2, *args, **kwargs)

'''返回值为布尔值数组'''
numpy.greater(x1, x2, *args, **kwargs)#判断x1>x2
numpy.greater_equal(x1, x2, *args, **kwargs)#>=
numpy.equal(x1, x2, *args, **kwargs)#==
numpy.not_equal(x1, x2, *args, **kwargs)#!=
numpy.less(x1, x2, *args, **kwargs)#<
numpy.less_equal(x1, x2, *args, **kwargs)#<=

'''一定范围判断数组相等'''
numpy.isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False)#一定范围内对应元素是否相等，返回布尔数组
numpy.allclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False)#判断一定范围数组是否相等返回布尔值（整个数组而言）
numpy.all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))#对比以上（没有一定范围）

'''排序（可以选择轴和那类元素）'''
numpy.sort(a[, axis=-1, kind='quicksort', order=None])

'''排序返回索引'''
numpy.argsort(a[, axis=-1, kind='quicksort', order=None])

'''指标排序'''
numpy.lexsort(keys[, axis=-1])

'''搜索'''
numpy.argmax(a[, axis=None, out=None])#>
numpy.argmin(a[, axis=None, out=None])#<
numpy.where(condition, [x=None, y=None])#输入条件
#搜索和对照很像，只是返回值变为索引
#有axis就是可以选择通过那个轴来搜索（就是选择从那个维度索引）

'''计数'''
numpy.count_nonzero(a, axis=None)#返回非零元素个数

'''找出数组中唯一值（有顺序）'''
numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)

'''返回包含函数元素的布尔值和索引值'''
numpy.in1d(ar1, ar2, assume_unique=False, invert=False)

'''返回交集元素及元素排序的索引'''
numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)

'''返回并集'''
numpy.union1d(ar1, ar2)

'''返回差集'''
numpy.setdiff1d(ar1, ar2, assume_unique=False)

'''返回异或'''
setxor1d(ar1, ar2, assume_unique=False)
```