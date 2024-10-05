import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import FinanceDataReader as fdr

# 데이터 불러오기
kospi_data = fdr.DataReader('KS11', '2020-01-01', '2024-12-31')  # KOSPI
sp500_data = fdr.DataReader('US500', '2020-01-01', '2024-12-31')  # S&P 500

# 필요 없는 컬럼 삭제 (KOSPI 데이터)
kospi_data.drop(['Change', 'UpDown', 'Comp', 'Amount', 'MarCap'], axis=1, inplace=True)

# 필요 없는 컬럼 삭제 (S&P 500 데이터에서 'Adj Close' 삭제)
sp500_data.drop(['Adj Close'], axis=1, inplace=True)

# 인덱스에서 Date를 열로 변환
kospi_data = kospi_data.reset_index()
sp500_data = sp500_data.reset_index()

# 자산 구분 열 추가
kospi_data['Asset'] = 'KOSPI'
sp500_data['Asset'] = 'S&P500'

# index 열 이름을 Date로 변경
sp500_data.rename(columns={'index': 'Date'}, inplace=True)

# KOSPI와 S&P 데이터를 합치기 (통일된 컬럼만 사용)
common_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset']
kospi_data = kospi_data[common_columns]
sp500_data = sp500_data[common_columns]

# 데이터 합치기
stock_data = pd.concat([kospi_data, sp500_data])

print(stock_data)
