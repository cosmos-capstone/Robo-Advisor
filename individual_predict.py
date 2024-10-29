import numpy as np
import pandas as pd
from keras import Model
from keras.src.layers import LSTM, Dense, Input, Embedding, Concatenate, Flatten
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import FinanceDataReader as fdr

def load_data(ticker, start, end):
    # 데이터 불러오기
    data = fdr.DataReader(ticker, start, end)
    
    # 인덱스를 Date 컬럼으로 변환 (reset_index를 통해 변환이 안 될 경우 대비)
    if data.index.name != 'Date':
        data.index.name = 'Date'
    data = data.reset_index()

    # 공통 컬럼 선택
    if 'Change' in data.columns:  # 한국 주식의 경우
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    elif 'Adj Close' in data.columns:  # 해외 주식의 경우
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return data

# 종목별 데이터 준비
kospi_data = load_data('KS11', '2020-01-01', '2024-12-31')  # KOSPI
sp500_data = load_data('US500', '2020-01-01', '2024-12-31')  # S&P 500
samsung_data = load_data('005930', '2020-01-01', '2024-12-31')  # Samsung
sk_hynix_data = load_data('000660', '2020-01-01', '2024-12-31')  # SK Hynix

# 종목 코드 할당 (각 종목에 고유 ID를 부여)
kospi_data['StockCode'] = 0
sp500_data['StockCode'] = 1
samsung_data['StockCode'] = 2
sk_hynix_data['StockCode'] = 3

# 데이터 합치기
all_data = pd.concat([kospi_data, sp500_data, samsung_data, sk_hynix_data])
all_data['Date'] = pd.to_datetime(all_data['Date'])
all_data.sort_values('Date', inplace=True)

# 스케일링
scaler = StandardScaler()
all_data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(all_data[['Open', 'High', 'Low', 'Close', 'Volume']])

# 학습 데이터 준비 함수
def prepare_data(data, seq_len=14, pred_days=1):
    sequences, stock_codes, targets = [], [], []
    
    for stock_code in data['StockCode'].unique():
        stock_data = data[data['StockCode'] == stock_code]
        stock_values = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        stock_codes_array = stock_data['StockCode'].values

        for i in range(seq_len, len(stock_values) - pred_days + 1):
            sequences.append(stock_values[i - seq_len:i])
            stock_codes.append(stock_codes_array[i])
            targets.append(stock_values[i + pred_days - 1][0])  # 'Open'을 예측

    return np.array(sequences), np.array(stock_codes), np.array(targets)

seq_len = 14
trainX, train_codes, trainY = prepare_data(all_data)
train_codes = train_codes.reshape(-1, 1)


# 모델 구성 (종목 코드 임베딩 포함)
def build_model(input_shape, num_stock_codes):
    # 가격 데이터 입력
    price_input = Input(shape=input_shape, name='price_input')
    lstm_out = LSTM(64, return_sequences=True)(price_input)
    lstm_out = LSTM(32, return_sequences=False)(lstm_out)

    # 종목 코드 임베딩 입력
    code_input = Input(shape=(1,), name='code_input')
    embedding = Embedding(input_dim=num_stock_codes, output_dim=4)(code_input)
    embedding = Flatten()(embedding)

    # 가격 데이터와 종목 코드 결합
    merged = Concatenate()([lstm_out, embedding])

    # 출력층
    output = Dense(1)(merged)
    
    model = Model(inputs=[price_input, code_input], outputs=output)
    return model

# 모델 학습
num_stock_codes = all_data['StockCode'].nunique()
model = build_model((seq_len, trainX.shape[2]), num_stock_codes)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 학습 (종목 코드와 가격 데이터를 함께 전달)
history = model.fit(
    {'price_input': trainX, 'code_input': train_codes},
    trainY,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 예측 예시 (KOSPI)
kospi_testX, kospi_test_codes, kospi_testY = prepare_data(kospi_data)
kospi_test_codes = kospi_test_codes.reshape(-1, 1)  # (batch_size, 1) 형태로 변환
predictions = model.predict({'price_input': kospi_testX, 'code_input': kospi_test_codes})

# 결과 복원 및 시각화
predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 4)))))[:, 0]
actual = scaler.inverse_transform(np.hstack((kospi_testY.reshape(-1, 1), np.zeros((kospi_testY.shape[0], 4)))))[:, 0]

plt.plot(actual, label="Actual Open Price")
plt.plot(predictions, label="Predicted Open Price", linestyle="--")
plt.title("KOSPI: Actual vs Predicted")
plt.xlabel("Days")
plt.ylabel("Open Price")
plt.legend()
plt.show()
