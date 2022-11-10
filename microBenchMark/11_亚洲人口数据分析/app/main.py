#!/usr/bin/env python
# coding: utf-8

# In[19]:


import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')
# 解决中文显示问题
plt.rc('font', family='Microsoft YaHei')


# > 1.数据预处理

# In[20]:


# 只读取第一行，获取列索引
total_data = np.array(
    open('./data/AsianPopulationData.csv', encoding='utf-8').readline()
)
total_data


# In[21]:


# 获取国家索引
country_index = np.array(
    open('./data/AsianPopulationData.csv', encoding='utf-8').readline()[:-1].split(',')
)
country_index


# In[22]:


# 读取人口数据
population_data = np.genfromtxt('./data/AsianPopulationData.csv', encoding='utf-8', delimiter=',', skip_header=1, dtype=np.str_)
# population_data
# 时间索引(获取每一行的第一个元素)
time_index = population_data[:, 0]
time_index


# > 2.计算2015年各个国家的人口数据

# In[23]:


year = '2015'
# 获取year年所有国家的人口数据
population_by_year = population_data[time_index == year]
# 提取前十个国家的数据进行显示
population_by_year = population_by_year[0][:11]
# print(population_by_year)


# - 数据以文本形式展示

# In[24]:


print('%s年各个国家人口数据：'% year)
print("-"*30)
for country_name, country_data in zip(country_index[1:], population_by_year[1:]):
    print('"%s"人口为：\t%s' % (country_name, country_data))


# - 数据以柱状图展示

# In[25]:


# 格式化数据
country_data = DataFrame(zip(country_index[1:], population_by_year[1:]), columns=["country", "num"])
country_data


# In[26]:


# sns.barplot(x=country_data["country"].index, y=country_data['num'].values)
# country_data["num"].index
# country_data["num"].values


# In[27]:


# bar宽度
bar_width = 0.3
# 设置画布大小
plt.figure(figsize=(18, 12))
# 每一个bar占0.3宽度
plt.bar(country_data["country"].values, country_data['num'].values, width=bar_width, alpha=0.7, label='国家', color='b')
# 显示图例
plt.legend(loc=1)

# 刻度字体大小
plt.tick_params(labelsize=16)
# 设置x轴刻度标签
# plt.xticks([ ix for ix in country_data["country"].values], rotation=16)
# 设置标题
plt.title('2015年各个国家的人口数据')
plt.xlabel('国家')
plt.ylabel('人口（个）')

# 显示网格
plt.grid(True, linestyle = "-.", linewidth = "3") 

plt.show()

