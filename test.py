import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import FinanceDataReader as fdr

# KOSPI 데이터 가져오기
kospi_data = fdr.DataReader('KS11', '2020-01-01', '2024-12-31')  # KOSPI

# SP500 데이터 가져오기
sp500_data = fdr.DataReader('US500', '2020-01-01', '2024-12-31')  # S&P 500

# 필요 없는 컬럼 삭제
kospi_data.drop(['Change', 'UpDown', 'Comp', 'Amount', 'MarCap'], axis=1, inplace=True)
sp500_data.drop(['Adj Close'], axis=1, inplace=True)

print(kospi_data.shape[0]) # 행 수
print(kospi_data.shape[1]) # 열 수 