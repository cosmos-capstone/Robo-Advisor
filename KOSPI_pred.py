import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import datetime

today = datetime.datetime.today().strftime('%Y-%m-%d')


kospi_data = fdr.DataReader('KS11', '2020-01-01', today)  # KOSPI

kospi_data.drop(['Change', 'UpDown', 'Comp', 'Amount', 'MarCap'], axis=1, inplace=True) # 필요 없는 컬럼 삭제

original_open = kospi_data['Open'].values
kospi_data = kospi_data.reset_index()
dates = pd.to_datetime(kospi_data['Date'])

cols = list(kospi_data)[1:6]  # 'Open', 'High', 'Low', 'Close', 'Volume' 열만 사용

# 데이터 정규화
kospi_data = kospi_data[cols].astype(float)
scaler = StandardScaler()
scaler = scaler.fit(kospi_data)
stock_data_scaled = scaler.transform(kospi_data)

# 학습 데이터 구성 (9:1로 나누지 않고 전체 사용)
seq_len = 365  # 과거 365일 데이터로 미래 예측
trainX = []
trainY = []

for i in range(seq_len, len(stock_data_scaled)):
    trainX.append(stock_data_scaled[i - seq_len:i, 0:stock_data_scaled.shape[1]])
    trainY.append(stock_data_scaled[i, 0])  # 다음 날의 'Open' 값을 예측

trainX, trainY = np.array(trainX), np.array(trainY)

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1))

model.summary()

learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

try:
    model.load_weights('./save_weights/kospi.weights.h5')
    print("Loaded model weights from disk")
except:
    print("No weights found, training model from scratch")
    history = model.fit(trainX, trainY, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
    model.save_weights('./save_weights/kospi.weights.h5')

    # plt.plot(history.history['loss'], label='Training loss')
    # plt.plot(history.history['val_loss'], label='Validation loss')
    # plt.legend()
    # plt.show()

# 미래 데이터 예측 (오늘 이후로 10일간 예측)
future_days = 10
last_sequence = stock_data_scaled[-seq_len:]  # 가장 최근의 1년 데이터를 가져옴

future_predictions = []

for _ in range(future_days):
    prediction = model.predict(np.expand_dims(last_sequence, axis=0))
    future_predictions.append(prediction[0, 0])  # 예측값 저장

    # prediction을 2차원 배열로 변환하여 시퀀스 업데이트
    predicted_sequence = np.repeat(prediction, last_sequence.shape[1], axis=-1)  # 예측값을 반복하여 차원 맞추기
    next_sequence = np.concatenate((last_sequence[1:], predicted_sequence), axis=0)
    last_sequence = next_sequence

# 정규화된 예측값을 원래 값으로 변환
mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], future_days, axis=0)
mean_values_pred[:, 0] = np.squeeze(future_predictions)
future_predicted_prices = scaler.inverse_transform(mean_values_pred)[:, 0]

# 예측된 미래 날짜 생성
future_dates = pd.date_range(start=dates.iloc[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')

# 결과 시각화
plt.figure(figsize=(14, 5))
plt.plot(dates, original_open, color='green', label='Original Open Price')
plt.plot(future_dates, future_predicted_prices, color='red', linestyle='--', label='Predicted Future Open Price')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Original and Predicted Future Open Price')
plt.legend()
plt.show()
