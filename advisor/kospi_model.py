import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import datetime

# 현재 날짜 설정
today = datetime.datetime.today().strftime('%Y-%m-%d')

# KOSPI 데이터 불러오기
kospi_data = fdr.DataReader('KS11', '2020-01-01', today)  # KOSPI

# 필요 없는 컬럼 삭제
kospi_data.drop(['Change', 'UpDown', 'Comp', 'Amount', 'MarCap'], axis=1, inplace=True)

# 원래 'Open' 데이터 저장
original_open = kospi_data['Open'].values
kospi_data = kospi_data.reset_index()
dates = pd.to_datetime(kospi_data['Date'])

# 'Open', 'High', 'Low', 'Close', 'Volume' 열만 사용
cols = list(kospi_data)[1:6]

# 데이터 정규화
kospi_data = kospi_data[cols].astype(float)
scaler = StandardScaler()
scaler = scaler.fit(kospi_data)
stock_data_scaled = scaler.transform(kospi_data)

# 학습 데이터 구성 (120일 후의 'Open' 값을 예측)
seq_len = 365  # 과거 365일 데이터로 미래 예측
future_days = 120  # 120일 후의 값을 예측

trainX = []
trainY = []

for i in range(seq_len, len(stock_data_scaled) - future_days):
    trainX.append(stock_data_scaled[i - seq_len:i, 0:stock_data_scaled.shape[1]])
    trainY.append(stock_data_scaled[i + future_days, 0])  # 120일 후의 'Open' 값을 예측

trainX, trainY = np.array(trainX), np.array(trainY)

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1))

model.summary()

# 학습률 설정 및 Adam 옵티마이저 사용
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

# 가중치 로드 또는 모델 학습
try:
    model.load_weights('./save_weights/kospi_120days.weights.h5')
    print("Loaded model weights from disk")
except:
    print("No weights found, training model from scratch")
    history = model.fit(trainX, trainY, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
    model.save_weights('./save_weights/kospi_120days.weights.h5')

    # 학습 결과 시각화
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

# 120일 후의 값을 예측
last_sequence = stock_data_scaled[-seq_len:]  # 가장 최근의 365일 데이터를 가져옴

# 예측 수행
prediction = model.predict(np.expand_dims(last_sequence, axis=0))

# 예측된 값을 원래 값으로 변환
mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], 1, axis=0)
mean_values_pred[:, 0] = np.squeeze(prediction)
future_predicted_price = scaler.inverse_transform(mean_values_pred)[:, 0]

# 예측된 날짜 생성 (120일 후의 날짜)
future_date = pd.date_range(start=dates.iloc[-1] + pd.Timedelta(days=future_days), periods=1, freq='B')
print(future_predicted_price)



# # 결과 시각화
# plt.figure(figsize=(14, 5))
# plt.plot(dates, original_open, color='green', label='Original Open Price')
# plt.plot(future_date, future_predicted_price, color='red', linestyle='--', label='Predicted 120 Days Future Open Price')
# plt.xlabel('Date')
# plt.ylabel('Open Price')
# plt.title('Original and Predicted 120 Days Future Open Price')
# plt.legend()
# plt.show()
