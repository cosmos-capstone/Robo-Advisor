import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from datetime import timedelta


# 데이터 불러오기
kospi_data = fdr.DataReader('KS11', '2020-01-01', '2024-12-31')  # KOSPI
sp500_data = fdr.DataReader('US500', '2020-01-01', '2024-12-31')  # S&P 500

# 필요 없는 컬럼 삭제
kospi_data.drop(['Change', 'UpDown', 'Comp', 'Amount', 'MarCap'], axis=1, inplace=True)
sp500_data.drop(['Adj Close'], axis=1, inplace=True)

# 인덱스에서 Date를 열로 변환
kospi_data = kospi_data.reset_index()
sp500_data = sp500_data.reset_index()
sp500_data.rename(columns={'index': 'Date'}, inplace=True)

# 자산별 데이터 처리 함수
def prepare_data(data):
    # 필요한 컬럼만 사용
    common_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    data = data[common_columns]

    # 날짜 데이터 분리
    dates = pd.to_datetime(data['Date'])

    # 학습용 데이터만 추출
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[cols].astype(float)

    # 데이터 정규화
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)

    return data_scaled, scaler, dates

# KOSPI와 S&P 데이터를 각각 준비
kospi_scaled, kospi_scaler, kospi_dates = prepare_data(kospi_data)
sp500_scaled, sp500_scaler, sp500_dates = prepare_data(sp500_data)

# 학습/테스트 데이터 분할 함수
def split_data(data_scaled, dates, seq_len=14, pred_days=1):
    n_train = int(0.9 * data_scaled.shape[0])
    
    # 학습 데이터
    train_data_scaled = data_scaled[0:n_train]
    train_dates = dates[0:n_train]

    # 테스트 데이터
    test_data_scaled = data_scaled[n_train:]
    test_dates = dates[n_train:]

    # LSTM 입력 형식으로 데이터 변환
    trainX, trainY, testX, testY = [], [], [], []
    
    for i in range(seq_len, n_train - pred_days + 1):
        trainX.append(train_data_scaled[i - seq_len:i, 0:train_data_scaled.shape[1]])
        trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, 0])

    for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
        testX.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
        testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, 0])

    return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY), test_dates

# KOSPI와 S&P 데이터 분리
kospi_trainX, kospi_trainY, kospi_testX, kospi_testY, kospi_test_dates = split_data(kospi_scaled, kospi_dates)
sp500_trainX, sp500_trainY, sp500_testX, sp500_testY, sp500_test_dates = split_data(sp500_scaled, sp500_dates)


# LSTM 모델 구성
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1))
    
    return model

# 모델 학습 함수
def train_or_load_model(trainX, trainY, asset_name):
    model = build_model((trainX.shape[1], trainX.shape[2]))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

    # 가중치 파일 경로
    weight_path = f'./save_weights/lstm_{asset_name}.weights.h5'

    # 가중치가 있으면 로드하고, 없으면 학습 수행
    try:
        model.load_weights(weight_path)
        print(f"Loaded model weights for {asset_name} from disk")
    except:
        print(f"No weights found for {asset_name}, training model from scratch")
        history = model.fit(trainX, trainY, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
        model.save_weights(weight_path)

        # 학습 손실 시각화
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title(f'{asset_name} Training Loss')
        plt.legend()
        plt.show()

    return model

# 모델 예측 함수
def predict(model, testX, testY, scaler):
    # 예측 수행
    prediction = model.predict(testX)

    # 예측 데이터 복원 (스케일 역변환)
    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
    mean_values_pred[:, 0] = np.squeeze(prediction)
    y_pred = scaler.inverse_transform(mean_values_pred)[:, 0]

    # 테스트 데이터 복원 (스케일 역변환)
    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)
    mean_values_testY[:, 0] = np.squeeze(testY)
    testY_original = scaler.inverse_transform(mean_values_testY)[:, 0]

    return y_pred, testY_original

# KOSPI 모델 학습 및 예측
kospi_model = train_or_load_model(kospi_trainX, kospi_trainY, 'KOSPI')
kospi_pred, kospi_testY_original = predict(kospi_model, kospi_testX, kospi_testY, kospi_scaler)

# S&P 500 모델 학습 및 예측
sp500_model = train_or_load_model(sp500_trainX, sp500_trainY, 'S&P500')
sp500_pred, sp500_testY_original = predict(sp500_model, sp500_testX, sp500_testY, sp500_scaler)
seq_len = 14


# # 결과 시각화
# plt.figure(figsize=(14, 5))


# # KOSPI 결과
# plt.subplot(1, 2, 1)
# plt.plot(kospi_dates, kospi_data['Open'], color='green', label='Original Open Price (KOSPI)')
# plt.plot(kospi_test_dates[seq_len:], kospi_testY_original, color='blue', label='Actual Open Price (KOSPI)')
# plt.plot(kospi_test_dates[seq_len:], kospi_pred, color='red', linestyle='--', label='Predicted Open Price (KOSPI)')
# plt.xlabel('Date')
# plt.ylabel('Open Price')
# plt.title('KOSPI: Original, Actual and Predicted Open Price')
# plt.legend()

# # S&P500 결과
# plt.subplot(1, 2, 2)
# plt.plot(sp500_dates, sp500_data['Open'], color='green', label='Original Open Price (S&P500)')
# plt.plot(sp500_test_dates[seq_len:], sp500_testY_original, color='blue', label='Actual Open Price (S&P500)')
# plt.plot(sp500_test_dates[seq_len:], sp500_pred, color='red', linestyle='--', label='Predicted Open Price (S&P500)')
# plt.xlabel('Date')
# plt.ylabel('Open Price')
# plt.title('S&P 500: Original, Actual and Predicted Open Price')
# plt.legend()

# plt.tight_layout()
# plt.show()



def multi_step_forecast(model, last_sequence, days_ahead, scaler):
    forecast = []
    sequence = last_sequence.copy()  # 마지막 시퀀스를 복사하여 사용
    
    for _ in range(days_ahead):
        # 예측 수행
        prediction = model.predict(sequence[np.newaxis, :, :])
        
        # 예측한 값의 차원을 맞춰서 `sequence`의 끝에 추가
        prediction = np.repeat(prediction, sequence.shape[1], axis=-1)  # (1, feature_dim) 형태로 변환
        
        # 현재 시퀀스의 끝에 추가하고, 첫 번째 값을 제거하여 업데이트
        sequence = np.append(sequence[1:], prediction, axis=0)
        
        # 예측한 값을 리스트에 저장 (역변환 필요)
        mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], 1, axis=0)
        mean_values_pred[:, 0] = prediction[0, 0]  # 예측된 Close 값
        y_pred_original = scaler.inverse_transform(mean_values_pred)[:, 0]
        
        forecast.append(y_pred_original[0])  # 예측한 값을 추가

    return forecast


# 마지막 시퀀스 선택 (KOSPI와 S&P 500 각각에 대해 마지막 학습 시퀀스 사용)
days_ahead = 14  # 예측할 날 수

# KOSPI 14일 예측
kospi_last_sequence = kospi_testX[-1]  # 마지막 시퀀스
kospi_forecast = multi_step_forecast(kospi_model, kospi_last_sequence, days_ahead, kospi_scaler)

# S&P500 14일 예측
sp500_last_sequence = sp500_testX[-1]  # 마지막 시퀀스
sp500_forecast = multi_step_forecast(sp500_model, sp500_last_sequence, days_ahead, sp500_scaler)


from datetime import datetime, timedelta

# 오늘 날짜로부터 14일 후까지의 날짜 생성
today = datetime.now()
forecast_dates = [today + timedelta(days=i) for i in range(1, days_ahead + 1)]

# 예측 결과 시각화
plt.figure(figsize=(14, 5))

# KOSPI 예측 결과
plt.subplot(1, 2, 1)
plt.plot(kospi_dates, kospi_data['Open'], color='green', label='Original Open Price (KOSPI)')
plt.plot(kospi_test_dates[seq_len:], kospi_testY_original, color='blue', label='Actual Open Price (KOSPI)')
plt.plot(kospi_test_dates[seq_len:], kospi_pred, color='red', linestyle='--', label='Predicted Open Price (KOSPI)')
plt.plot(forecast_dates, kospi_forecast, color='purple', label='Forecasted Price (KOSPI)')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('KOSPI: Original, Actual, Predicted and Forecasted Open Price')
plt.legend()

# S&P500 예측 결과
plt.subplot(1, 2, 2)
plt.plot(sp500_dates, sp500_data['Open'], color='green', label='Original Open Price (S&P500)')
plt.plot(sp500_test_dates[seq_len:], sp500_testY_original, color='blue', label='Actual Open Price (S&P500)')
plt.plot(sp500_test_dates[seq_len:], sp500_pred, color='red', linestyle='--', label='Predicted Open Price (S&P500)')
plt.plot(forecast_dates, sp500_forecast, color='purple', linestyle='--', label='Forecasted Price (S&P500)')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('S&P 500: Original, Actual, Predicted and Forecasted Open Price')
plt.legend()

plt.tight_layout()
plt.show()
