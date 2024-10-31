import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from keras.src.models import load_model

K_stock_ticker = '379810'
A_stock_ticker = 'QQQM'


kospi_data = fdr.DataReader(K_stock_ticker, '2020-01-01', '2024-12-31')  
sp500_data = fdr.DataReader(A_stock_ticker, '2016-01-01', '2024-12-31')  
print(kospi_data)


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

# 학습 데이터를 시퀀스 형식으로 변환하는 함수 (예측 대상은 전체 데이터)
def create_sequences(data_scaled, seq_len=14):
    sequences = []
    targets = []

    for i in range(seq_len, len(data_scaled)):
        sequences.append(data_scaled[i - seq_len:i])
        targets.append(data_scaled[i, 0])  # 'Open'을 예측 대상으로 설정

    return np.array(sequences), np.array(targets)

# 전체 데이터로 학습 시퀀스 생성
kospi_trainX, kospi_trainY = create_sequences(kospi_scaled)
sp500_trainX, sp500_trainY = create_sequences(sp500_scaled)

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

    # 전체 모델 파일 경로
    model_path = f'./save_models/lstm_{asset_name}.h5'

    # 모델이 있으면 로드하고, 없으면 학습 수행
    try:
        model = load_model(model_path)
        print(f"Loaded full model for {asset_name} from disk")
    except:
        print(f"No saved model found for {asset_name}, training model from scratch")
        history = model.fit(trainX, trainY, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
        model.save(model_path)  # 전체 모델 저장

        # 학습 손실 시각화
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title(f'{asset_name} Training Loss')
        plt.legend()
        plt.show()

    return model

# 다단계 예측 함수
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
        mean_values_pred[:, 0] = prediction[0, 0]  # 예측된 Open 값
        y_pred_original = scaler.inverse_transform(mean_values_pred)[:, 0]
        
        forecast.append(y_pred_original[0])  # 예측한 값을 추가

    return forecast
# KOSPI 모델 학습 및 예측
kospi_model = train_or_load_model(kospi_trainX, kospi_trainY, K_stock_ticker)
kospi_last_sequence = kospi_trainX[-1]  # 마지막 시퀀스
days_ahead = 14  # 예측할 날 수

# last sequence를 오늘 날짜로 바꿔서 
kospi_forecast = multi_step_forecast(kospi_model, kospi_last_sequence, days_ahead, kospi_scaler)

# S&P 500 모델 학습 및 예측
sp500_model = train_or_load_model(sp500_trainX, sp500_trainY, A_stock_ticker)
sp500_last_sequence = sp500_trainX[-1]  # 마지막 시퀀스
sp500_forecast = multi_step_forecast(sp500_model, sp500_last_sequence, days_ahead, sp500_scaler)

# 오늘 날짜로부터 14일 후까지의 날짜 생성
today = datetime.now()
forecast_dates = [today + timedelta(days=i) for i in range(1, days_ahead + 1)]
# 예측 날짜 형식 변경 및 소수점 반올림 적용
def format_forecast_results(dates, forecasts):
    formatted_results = [(date.strftime('%Y-%m-%d'), round(forecast, 2)) for date, forecast in zip(dates, forecasts)]
    return formatted_results

# 오늘 날짜로부터 14일 후까지의 날짜 생성
today = datetime.now()
forecast_dates = [today + timedelta(days=i) for i in range(1, days_ahead + 1)]

# 예측 결과 포맷팅
kospi_forecast_results = format_forecast_results(forecast_dates, kospi_forecast)
sp500_forecast_results = format_forecast_results(forecast_dates, sp500_forecast)

print(kospi_forecast_results, sp500_forecast_results)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(kospi_dates, kospi_data['Open'], color='green', label='Original Open Price (KOSPI)')
plt.plot(forecast_dates, kospi_forecast, color='purple', label='Forecasted Price (KOSPI)')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('KOSPI: Original and Forecasted Open Price')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sp500_dates, sp500_data['Open'], color='green', label='Original Open Price (S&P500)')
plt.plot(forecast_dates, sp500_forecast, color='purple', linestyle='--', label='Forecasted Price (S&P500)')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('S&P 500: Original and Forecasted Open Price')
plt.legend()

plt.tight_layout()
plt.show()
