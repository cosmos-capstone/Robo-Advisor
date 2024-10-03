import FinanceDataReader as fdr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 데이터 수집 예시
kospi = fdr.DataReader('KS11', '2020-01-01', '2024-12-31')  # KOSPI
snp = fdr.DataReader('US500', '2020-01-01', '2024-12-31')   # S&P 500
bond_etf = fdr.DataReader('114100', '2020-01-01', '2024-12-31')  # 국고채 ETF

print(kospi)

# 데이터 병합
data = pd.concat([kospi['Close'], snp['Close'], bond_etf['Close']], axis=1)
data.columns = ['KOSPI', 'S&P', 'Bond_ETF']

# 그래프 그리기
# plt.figure(figsize=(6, 3))
# plt.plot(data.index, data['KOSPI'], label='KOSPI', marker='o')
# # plt.plot(data.index, data['S&P'], label='S&P 500', marker='x')
# # plt.plot(data.index, data['Bond_ETF'], label='Bond ETF', marker='s')

# # 제목 및 라벨 설정
# plt.title('KOSPI, S&P 500, and Bond ETF Closing Prices')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)  # x축 라벨을 회전하여 가독성 개선
# plt.tight_layout()
# plt.show()

# 스케일링
# scaler = MinMaxScaler()
# data_scaled = scaler.fit_transform(data)


# import numpy as np

# def create_dataset(data, time_steps=30):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:i + time_steps])
#         y.append(data[i + time_steps])
#     return np.array(X), np.array(y)

# X, y = create_dataset(data_scaled)


# import tensorflow as tf
# from keras import Sequential
# from keras.src.layers import LSTM, Dense

# # 모델 생성
# model = Sequential([
#     LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
#     LSTM(32, activation='relu'),
#     Dense(16),
#     Dense(y.shape[1])
# ])

# model.compile(optimizer='adam', loss='mse')
# model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)


# predicted = model.predict(X)
# from scipy.optimize import minimize

# # 샤프 비율만 반환하도록 수정
# def calculate_sharpe(weights, returns):
#     port_return = np.sum(weights * returns.mean()) * 252
#     port_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
#     sharpe_ratio = port_return / port_risk
#     # 샤프 비율은 음수를 최소화하므로 최대화를 위해 -로 반환
#     return -sharpe_ratio

# # 최적화 함수 호출 시 목적 함수를 변경
# def optimize_portfolio(returns):
#     num_assets = len(returns.columns)
#     args = (returns,)
#     constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
#     bounds = tuple((0, 1) for asset in range(num_assets))
#     result = minimize(calculate_sharpe, num_assets*[1./num_assets], args=args, 
#                       method='SLSQP', bounds=bounds, constraints=constraints)
#     return result.x


# # 예측 데이터로 최적 포트폴리오 구성
# optimal_weights = optimize_portfolio(pd.DataFrame(predicted, columns=data.columns))
# print("최적의 포트폴리오 구성 비중: ", optimal_weights)
