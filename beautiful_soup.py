from bs4 import BeautifulSoup as bs
import requests
import numpy as np
import pandas as pd
from keras.datasets import imdb
import pickle

import yfinance

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

import numpy as np


#가설 > 기사에서 많이 다루는 회사들은, 유의미한 주가 상승이 있을 것이다.
#가설 > 2015년의 재무재표는 2024년의 주식 가격에 영향을 줄 것이다.
#검증 > 3000개의 나스닥 회사들을 가져와서, (2024의 주가/2015년의 주가)  / (2024년의 나스닥 주가/2015년의 나스닥 주가) > 1 로 이진 분석
#sigmoid 함수를 통해, 

# def vectorize_sequence(seq, dimension=10000) :
#     #크기가 (len(seq), dimension )이고 모든 원소가 0인 행렬 
#     result = np.zeros((len(seq), dimension ))
    
#     for i, s in enumerate(seq) :
#         result[i, s] = 1. #특정 인덱스의 위치를 1로 만든다.
        
#     return result

# print(train_data)
# # 훈련 데이터 벡터화 
# x_train = vectorize_sequence(train_data)
# print(x_train)
# x_test = vectorize_sequence(test_data)

# # 레이블 벡터화 
# y_train = np.asarray(train_labels).astype('float32')
# y_test = np.asarray(test_labels).astype('float32')

# from keras import models
# from keras import layers

# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])


# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]

# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]

# model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['acc'])

# history = model.fit(partial_x_train,
#                    partial_y_train,
#                    epochs=20,
#                    batch_size=512,
#                    validation_data=(x_val, y_val))



# history_dict = history.history
# history_dict.keys()
# print(history_dict)

with open('nasdaq_company_code.pkl', 'rb') as file:
    nasdaq_company_code = pickle.load(file)
    
    
for i in range(0,1):    
    msft = yfinance.Ticker(nasdaq_company_code[i])
    print(msft.income_stmt.iloc[datetime(2022,12,31)])
#     print(msft.quarterly_income_stmt)
# # - balance sheet
#     print(msft.balance_sheet)
#     print(msft.quarterly_balance_sheet)
# # - cash flow statement
#     print(msft.cashflow)
#     print(msft.quarterly_cashflow)



#==================================================

# newslists =[]
# wordslists =[]
# page = requests.get(f"https://www.nytimes.com/section/business/dealbook?page=10")
# soup = bs(page.text, "html.parser")

# elements = soup.find_all("h3","css-1kv6qi e15t083i0")



# for element in elements: 
#     newslists.append(element)      
   
# for newslist in newslists:
#     words = str(newslist)[33: -5].split()
#     for word in words:
#         wordslists.append(word)




# count={}
# for i in wordslists:
#     try: count[i] += 1
#     except: count[i]=1
    
# print(sorted(count.items(), key= lambda item:item[1]))

