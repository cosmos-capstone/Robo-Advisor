import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import FinanceDataReader as fdr

# 모든 한국 주식 종목 데이터를 불러오기
krx = fdr.StockListing('KRX')
stock_data_list = []

# 모든 종목의 데이터를 수집하여 리스트에 저장
for code, name in zip(krx['Code'], krx['Name']):
    try:
        data = fdr.DataReader(code, '2023-01-01', '2024-12-31')
        data['Code'] = code  # 종목 코드 추가
        data['Name'] = name  # 종목 이름 추가
        stock_data_list.append(data)
    except:
        pass  # 데이터가 없는 종목은 건너뜁니다.

# 모든 종목 데이터 합치기
stock_data = pd.concat(stock_data_list)

# 필요 없는 컬럼 삭제 후 인덱스 초기화
stock_data.drop(['Change'], axis=1, inplace=True)
stock_data = stock_data.reset_index()

# 원핫 인코딩 적용 (Code 열에 대해)
stock_data = pd.get_dummies(stock_data, columns=['Code'])

# 인덱스를 초기화하고 날짜 데이터 분리
dates = pd.to_datetime(stock_data['Date'])
cols = list(stock_data.columns.difference(['Date', 'Name']))

# 'Open' 데이터를 따로 저장
original_open = stock_data['Open'].values

# 학습용 데이터만 추출
stock_data = stock_data[cols].astype(float)

# 데이터 정규화
scaler = StandardScaler()
scaler = scaler.fit(stock_data)
stock_data_scaled = scaler.transform(stock_data)

# 학습/테스트 데이터 분할
n_train = int(0.9 * stock_data_scaled.shape[0])
train_data_scaled = stock_data_scaled[:n_train]
train_dates = dates[:n_train]

test_data_scaled = stock_data_scaled[n_train:]
test_dates = dates[n_train:]

# LSTM 입력 형식으로 데이터 변환
pred_days = 1  # 예측 주기
seq_len = 14   # 시퀀스 길이
input_dim = len(cols)  # 입력 차원

trainX, trainY = [], []
testX, testY = []

for i in range(seq_len, n_train - pred_days + 1):
    trainX.append(train_data_scaled[i - seq_len:i])
    trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, 0])

for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
    testX.append(test_data_scaled[i - seq_len:i])
    testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, 0])

trainX, trainY = np.array(trainX), np.array(trainY)
testX, testY = np.array(testX), np.array(testY)

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
    model.load_weights('./save_weights/lstm.weights.h5')
    print("Loaded model weights from disk")
except:
    print("No weights found, training model from scratch")
    history = model.fit(trainX, trainY, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
    model.save_weights('./save_weights/lstm.weights.h5')
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

# 특정 종목 예측 함수
def predict_stock(stock_code):
    # 해당 종목의 원핫 인코딩된 열을 찾습니다
    stock_cols = [col for col in cols if stock_code in col]
    if not stock_cols:
        print(f"Error: '{stock_code}'에 대한 데이터가 없습니다.")
        return
    
    # 예측 수행
    prediction = model.predict(testX)
    
    # 예측 복원 (스케일 역변환)
    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
    mean_values_pred[:, 0] = np.squeeze(prediction)
    y_pred = scaler.inverse_transform(mean_values_pred)[:, 0]

    # 결과 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(test_dates[seq_len:], y_pred, color='red', linestyle='--', label=f'Predicted Open Price ({stock_code})')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title(f'Predicted Open Price for {stock_code}')
    plt.legend()
    plt.show()

# 예측 예제
predict_stock('005930')  # 예: 삼성전자
