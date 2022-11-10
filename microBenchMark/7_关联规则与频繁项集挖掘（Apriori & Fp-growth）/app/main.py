# Importing the libraries
import pandas as pd
from efficient_apriori import apriori as apriori
from pyfpgrowth import pyfpgrowth

'''
例子
lists = [
    ['啤酒', '牛奶', '可乐'],
    ['尿不湿', '啤酒', '牛奶', '橙汁'],
    ['啤酒', '尿不湿'],
    ['啤酒', '可乐', '尿不湿'],
    ['啤酒', '牛奶', '可乐']
]
'''

if __name__ == '__main__':
    # Data Preprocessing
    dataset = pd.read_csv('./data/Market_Basket_Optimisation.csv', header=None)

    # Getting the list of transactions from the dataset
    transactions = []
    for i in range(0, dataset.shape[0]):
        arr = dataset.iloc[i].dropna()
        transactions.append(arr.tolist())

    print(transactions)

    # Apriori算法
    print('========Apriori算法开始========')
    itemsets, rules = apriori(transactions,
                              min_support=0.6,
                              min_confidence=1)

    print(itemsets)
    print('======')
    print(rules)
    print('========Apriori算法结束========')

    # Fp-growth 算法
    print('========Fp-growth算法开始========')
    patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)  # 频数删选  频数大于2
    rules = pyfpgrowth.generate_association_rules(patterns, 0.6)  # 置信度(条件概率)删选

    print(patterns)
    print('======')
    print(rules)
    print('========Fp-growth算法结束========')
