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

# 필요 없는 컬럼 삭제
kospi_data.drop(['Change', 'UpDown', 'Comp', 'Amount', 'MarCap'], axis=1, inplace=True)
sp500_data.drop(['Adj Close'], axis=1, inplace=True)
kospi_data = kospi_data.reset_index()
sp500_data = sp500_data.reset_index()
# 자산 구분 열 추가
kospi_data['Asset'] = 'KOSPI'
sp500_data['Asset'] = 'S&P500'

# KOSPI와 S&P 데이터를 합치기
stock_data = pd.concat([kospi_data, sp500_data])

# 원핫 인코딩 적용 (Asset 열에 대해)
stock_data = pd.get_dummies(stock_data, columns=['Asset'])
print(stock_data)
# 원래의 'Open' 데이터를 저장
original_open = stock_data['Open'].values

# 인덱스를 초기화하고 날짜 데이터 분리
stock_data = stock_data.reset_index()
dates = pd.to_datetime(stock_data['Date'])

# 입력 변수 설정 (원핫 인코딩된 자산 정보 포함)
cols = list(stock_data)[1:6]  # 'Open', 'High', 'Low', 'Close', 'Volume' + Asset(KOSPI, S&P500)

# 학습용 데이터만 추출
stock_data = stock_data[cols].astype(float)

# 데이터 정규화
scaler = StandardScaler()
scaler = scaler.fit(stock_data)
stock_data_scaled = scaler.transform(stock_data)

# 학습/테스트 데이터 분할
n_train = int(0.9 * stock_data_scaled.shape[0])
train_data_scaled = stock_data_scaled[0:n_train]
train_dates = dates[0:n_train]

test_data_scaled = stock_data_scaled[n_train:]
test_dates = dates[n_train:]

# LSTM 입력 형식으로 데이터 변환
pred_days = 1  # 예측 주기
seq_len = 14   # 시퀀스 길이 (이전 14일간 데이터를 사용)
input_dim = len(cols)  # 입력 차원 (자산 구분 포함)

trainX = []
trainY = []
testX = []
testY = []

for i in range(seq_len, n_train - pred_days + 1):
    trainX.append(train_data_scaled[i - seq_len:i, 0:train_data_scaled.shape[1]])
    trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, 0])

for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
    testX.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
    testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, 0])

trainX, trainY = np.array(trainX), np.array(trainY)
testX, testY = np.array(testX), np.array(testY)

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1))  # 출력 뉴런 수를 1로 설정 (예측할 값이 하나)

model.summary()

# 학습률 설정 및 Adam 옵티마이저 사용
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

# 가중치 로드 또는 모델 학습
try:
    model.load_weights('./save_weights/lstm.weights.h5')
    print("Loaded model weights from disk")
except:
    print("No weights found, training model from scratch")
    # 모델 학습
    history = model.fit(trainX, trainY, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
    model.save_weights('./save_weights/lstm.weights.h5')

    # 학습 결과 시각화
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

# 예측 수행
prediction = model.predict(testX)
print(prediction.shape, testY.shape)

# 예측 데이터 복원 (스케일 역변환)
mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
mean_values_pred[:, 0] = np.squeeze(prediction)
y_pred = scaler.inverse_transform(mean_values_pred)[:, 0]

# 테스트 데이터 복원 (스케일 역변환)
mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)
mean_values_testY[:, 0] = np.squeeze(testY)
testY_original = scaler.inverse_transform(mean_values_testY)[:, 0]
print(testY_original.shape)

# 결과 시각화
plt.figure(figsize=(14, 5))
plt.plot(dates, original_open, color='green', label='Original Open Price')
plt.plot(test_dates[seq_len:], testY_original, color='blue', label='Actual Open Price')
plt.plot(test_dates[seq_len:], y_pred, color='red', linestyle='--', label='Predicted Open Price')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Original, Actual and Predicted Open Price')
plt.legend()
plt.show()
