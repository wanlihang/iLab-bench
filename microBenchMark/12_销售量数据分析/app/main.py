#!/usr/bin/env python
# coding: utf-8

# ### 案例场景：每个销售型公司都有一定的促销费用，促销费用可以带来销售量的增加，当给出一定的销售费用，预计可以带来多大的商品销售量？

# In[1]:


import re
import numpy
from sklearn import linear_model
from matplotlib import pyplot as plt

# #### 1.导入数据

# In[2]:


fn = open('./data/data.txt', 'r')
all_data = fn.readlines()
fn.close()

# In[3]:


all_data

# #### 2.数据预处理

# In[4]:


x = []
y = []
for single_data in all_data:
    # 将x,y数据分割成列表形式
    tmp_data = re.split('\t|\n', single_data)
    x.append(float(tmp_data[0]))
    y.append(float(tmp_data[1]))
# 将列类型的数据转换成数组类型的数据
x = numpy.array(x).reshape([100, 1])
y = numpy.array(y).reshape([100, 1])

# In[5]:


# x
y

# #### 3.数据分析

# - 通过散点图来选择需要使用的模型

# In[6]:


plt.scatter(x, y)
plt.show()

# _通过散点图发现x,y呈明显的线性关系。初步判断可以选择线性回归进行模型拟合_

# #### 4.数据建模

# - 使用sklearn中的线性回归模型实现

# In[7]:


# 创建模型对象
model = linear_model.LinearRegression()
# 将x,y分别作为自变量和因变量输入模型进行训练
model.fit(x, y)

# #### 5.模型评估

# - 模型拟合的校验和评估

# In[8]:


# 获取模型的自变量系数
model_coef = model.coef_
# 获取模型的截距
model_intercept = model.intercept_
# 获取模型的决定系数R方
r2 = model.score(x, y)

# #### 6.销售预测

# - 给出促销费用，预测销售量

# In[9]:


# 促销费用
promotion_cost = 100000
promotion_cost = numpy.array(promotion_cost)
pre_y = model.predict(promotion_cost.reshape(-1, 1))

# In[10]:


pre_y[0][0]

# _由预测值可以得出，假如促销费用是10万，那么就可以有22万的销售量_

# In[10]:
