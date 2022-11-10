#!/usr/bin/env python
# coding: utf-8

# # 美国航班的准点数据分析

# In[6]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# In[7]:


# 航班数据的链接网址
link = 'https://projects.fivethirtyeight.com/flights/'

# In[8]:


# 查看本地dada目录中的数据文件（项目中的实验数据都在data目录中）
# win中
# !dir "../data/"
# linux中
# get_ipython().system('ls "../data"')


# In[9]:


# 从本地读取航班数据
flights_df = pd.read_csv("./data/usa_flights.csv")

# In[10]:


# 查看前五行数据
flights_df.head()

# ## 1.查看飞机延误时间最长的前10名

# In[11]:


flights_df.sort_values('arr_delay', ascending=False)[:10]

# **可得出初步结论，航空公司AA、DL的延误频次要比其他航空公司高，且AA航空公司比Dl航空公司高。如果有急事出行的话，可以尽量避开这两家航班公司。**

# ## 2.计算延误和没有延误的飞机所占的比例

# In[12]:


# 添加一列判断航班是否延误
flights_df['delayed'] = flights_df['arr_delay'].apply(lambda x: x > 0)
# 查看前五行数据
flights_df.head()

# In[13]:


# 查看延误飞机的数量（False：非延误  True：延误）
delay_data = flights_df['delayed'].value_counts()
delay_data

# In[14]:


# 计算延误航班所占的比例
delay_data[1] / (delay_data[0] + delay_data[1])

# **由此可得出在美国的所有航班中，有48%的航班都发生过延误。**

# In[15]:


# 计算每一个航空公司延误的情况
delay_group = flights_df.groupby(['unique_carrier', 'delayed'])
df_delay = delay_group.size().unstack()
df_delay

# **由上表可以得出，大部分航空公司的准点次数要大于延误次数**

# ### 图形展示

# In[16]:


import matplotlib.pyplot as plt

# In[17]:


# barh 柱状图
# stacked=True 横向展示
# figsize=[16, 6] 宽16，高6
# colormap='winter' 使用winter色
df_delay.plot(kind='barh', stacked=True, figsize=[16, 6], colormap='winter')
plt.show()

# **由图形可以更直观的看出，飞行次数最多的公司是WN。虽然Dl公司的延时时间是相对比较长的，但他的延误次数却相比与其他公司要低。其次AS、VX两家小型航空公司的延误次数也比较低。**

# #### 4.透视表

# In[18]:


# index 索引
# columns 列名
# values 要显示的值
# aggfunc 聚合函数
flight_by_carrier = flights_df.pivot_table(index='flight_date', columns='unique_carrier', values='flight_num',
                                           aggfunc='count')
flight_by_carrier

# **由透视表我们可以看出，每天航班次数最多的公司是WN，其次是DL和EV。由此可猜测他们几家应该是大型航空公司。**

# In[18]:
