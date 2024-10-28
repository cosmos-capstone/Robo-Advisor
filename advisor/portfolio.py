import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from get_data import FDRData
from KOSPI_pred import predict_future_kospi
# 날짜 설정
today = date.today()
start_date = '2020-02-28'
end_date = today

tickers = ['KS11', '148070']  # 한국 주식, 한국 채권 ETF

fdr_data = FDRData(tickers=tickers, period=(start_date, end_date))
fdr_price = fdr_data.get_price() 
fdr_price.columns = ['kr_stock','kr_bond']
fdr_price = fdr_price.dropna()

predicted_price, predicted_date = predict_future_kospi(future_days=120)

# 자산 별 수익률 계산
dummy = 117000.0 # 채권 가격 예측값으로 대체할 것
# `predicted_date`와 `predicted_price`를 `fdr_price`에 추가
future_date_df = pd.DataFrame({'kr_stock': [predicted_price[0]], 'kr_bond': [dummy]}, index=predicted_date)
# `fdr_price`에 미래 예측 데이터를 추가
fdr_price = pd.concat([fdr_price, future_date_df])
price_rate = fdr_price / fdr_price.iloc[0]
return_rate = ((fdr_price.iloc[-1]) / fdr_price.iloc[-2] - 1) * 100
print(return_rate) # class(type) = pandas.core.series.Series 
return_rate[1] += 3.0 # 채권 연 평균 이자율로 대체


# # 자산 별 상관관계 히트맵 시각화
# plt.title("Correlation of all asset's percent change")
# sns.heatmap(fdr_price.pct_change().corr(), cmap='Blues', linewidth=0.2, annot=True)
# plt.show()


# 최적 포트폴리오 계산 부분 시작
port_ratios = []
port_returns = np.array([])
port_risks = np.array([])

# 포트폴리오 비율 조합 10,000개 생성 및 연평균 수익률/위험 계산
for i in range(10000): 
    # 포트폴리오 비율
    port_ratio = np.random.rand(len(fdr_price.columns)) # 2가지 자산에 대해 랜덤 비율 생성
    port_ratio /= port_ratio.sum() # 합계가 1인 랜덤 실수
    port_ratios.append(port_ratio)
    
    # 연 평균 수익률 계산
    total_return_rate = (fdr_price.iloc[-1] / fdr_price.iloc[0])  # 총 수익률(%)
    annual_avg_rr = total_return_rate ** (1 / 4) - 1  # 연평균 수익률 (4년간 데이터)
    port_return = np.dot(port_ratio, annual_avg_rr)  # 연 평균 포트폴리오 수익률 = 연 평균 수익률과 포트폴리오 비율의 행렬곱
    port_returns = np.append(port_returns, port_return)
    
    # 연간 수익률 공분산 계산
    annual_cov = fdr_price.pct_change().cov() * len(fdr_price) / 4  # 일별 수익률 공분산을 연간으로 변환
    port_risk = np.sqrt(np.dot(port_ratio.T, np.dot(annual_cov, port_ratio)))  # 포트폴리오 위험
    port_risks = np.append(port_risks, port_risk)

# 샤프 비율에 따른 포트폴리오 정렬
sorted_shape_idx = np.argsort(port_returns / port_risks)
sorted_risk_idx = np.argsort(port_risks)

# # 포트폴리오 시각화
# plt.figure(figsize=(12,6))
# sns.scatterplot(x=port_risks, y=port_returns, c=port_returns / port_risks, cmap='cool', alpha=0.85, s=20)
# sns.scatterplot(x=port_risks[sorted_shape_idx[-1:]], y=port_returns[sorted_shape_idx[-1:]], color='r', marker='^', s=500)
# sns.scatterplot(x=port_risks[sorted_risk_idx[:1]], y=port_returns[sorted_risk_idx[:1]], color='b', marker='v', s=500)

# plt.title('Return per unit risk (Sharpe Ratio)')
# plt.xlabel('Risk (Volatility)')
# plt.ylabel('Return')
# plt.show()

# 최적 포트폴리오 비율 시각화
port_df = pd.DataFrame(port_ratios)
sorted_port_df = port_df.iloc[sorted_shape_idx[::-1]]  # 역순
sorted_port_df.columns = fdr_price.columns

plt.figure(figsize=(12,4))
plt.stackplot(np.arange(1, len(sorted_port_df) + 1, 1), np.array(sorted_port_df.T), labels=sorted_port_df.columns)
plt.xlim(0, 10000)
plt.legend(bbox_to_anchor=(1.12, 0.95))
plt.xlabel('Ranking of Sharpe Ratio')
plt.ylabel('Portfolio Ratio')
plt.title('Ranking of Optimal Portfolios by Sharpe Ratio')
plt.show()

# 최적 포트폴리오 비율 출력
optimal_portfolio = pd.Series(sorted_port_df.iloc[0], index=sorted_port_df.columns)
print(f'최적의 포트폴리오 비율 : \n{optimal_portfolio}')
