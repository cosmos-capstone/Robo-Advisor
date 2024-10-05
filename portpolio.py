from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import yfinance as yf
import time


class YFinance():
    def __init__(self, tickers, period):
        self.period = period
        
        self.df_price = pd.DataFrame()
        self.df_dividends = pd.DataFrame()
        for ticker in tickers:
            print(f'[INFO] 데이터 불러오는 중 .. (ticker : {ticker})')
            full_data = yf.Ticker(ticker).history(start=self.period[0], end=self.period[1])
            self.df_price[ticker] = full_data['Close']
            self.df_dividends[ticker] = full_data['Dividends']
            time.sleep(0.5)
        print(f'[INFO] 데이터셋 구성 완료')
            
    def get_price(self):
        return self.df_price
        
    def get_dividends(self):
        return self.df_dividends
    

tickers = ['SPY','^KS11','^TNX','148070.KS'] # 미국 주식, 한국 주식, 미국 채권, 한국 채권
start_date = '2020-02-28'
end_date = '2024-02-28'

y_finance = YFinance(tickers=tickers, period=(start_date, end_date))

yf_price = y_finance.get_price() # 주가
yf_dividends = y_finance.get_dividends() # 배당금

print('-------------------------------\n',yf_price.shape, yf_dividends.shape)


# yf_price = yf_price.dropna()
yf_price.columns = ['us_stock','kr_stock','us_bond','kr_bond']

# yf_dividends = yf_dividends.dropna()
yf_dividends.columns = yf_price.columns

print(yf_price, yf_dividends)

# # price_rate = yf_price/yf_price.iloc[0] # 기준일(2011-10-20) 대비 증감

# # plt.figure(figsize=(12,4))
# # sns.lineplot(data=price_rate, linewidth=0.85)
# # plt.ylim((0, price_rate.max().max()))
# # plt.title('Increase/decrease rate compared to the base date')
# # plt.show()

# plt.figure(figsize=(12,8))

# pcc = yf_price.pct_change().iloc[1:,:] # 첫째 날 데이터 제거(NaN)

# # for i in range(4):

# #     data = pcc.iloc[:,i]
# #     plt.subplot(int(f'22{i+1}'))
# #     sns.lineplot(data=data, linewidth=0.85, alpha=0.7)
# #     inc_rate = (data > 0).sum() / len(data) * 100
# #     plt.title(f'< {data.name} : Increase rate {inc_rate:.2f}% >')
# #     plt.axhline(y=0, color='r', linestyle='--', linewidth=0.7, alpha=0.9)

# # plt.suptitle('Percent change of each asset')
# # plt.tight_layout()
# # plt.show()

# # 총 수익률
# # return_rate = ((yf_price.iloc[-1] + yf_dividends.sum()) / yf_price.iloc[0] - 1) * 100
# # return_rate

# plt.figure(figsize=(9,4))
# # bars = sns.barplot(x=return_rate.index, y=return_rate.values, color='Blue', alpha=0.3)
# # for p in bars.patches:
# #     bars.annotate(f'{p.get_height():.1f}%',
# #                   (p.get_x() + p.get_width() / 2., p.get_height()),
# #                    ha = 'center', va='center',
# #                    xytext = (0,9),
# #                    textcoords = 'offset points')
    
# # plt.title(f'Return rate of total period (%) - {(yf_price.index[-1] - yf_price.index[0]).days} days')
# # plt.show()

# plt.title("Correlation of all asset's percent change")
# sns.heatmap(yf_price.pct_change().corr(), cmap='Blues', linewidth=0.2, annot=True)
# plt.show()

# import numpy as np

# port_ratios = []
# port_returns = np.array([])
# port_risks = np.array([])
# for i in range(10000): # 포트폴리오 비율 조합 1000개
#     # 포트폴리오 비율
#     port_ratio = np.random.rand(len(yf_price.columns)) # 4가지 랜덤 실수 조합
#     port_ratio /= port_ratio.sum() # 합계가 1인 랜덤 실수
#     port_ratios.append(port_ratio)
    
#     # 연 평균 수익률
#     total_return_rate = (yf_price.iloc[-1] + yf_dividends.sum()) / yf_price.iloc[0] # 배당금 합산 총 수익률(%)
#     annual_avg_rr = total_return_rate ** (1/10) # 연 (기하)평균 수익률(%)
#     port_return = np.dot(port_ratio, annual_avg_rr-1) # 연 평균 포트폴리오 수익률 = 연 평균 수익률과 포트폴리오 비율의 행렬곱
#     port_returns = np.append(port_returns, port_return)
    
#     # 연간 수익률 공분산
#     annual_cov = yf_price.pct_change().cov() * len(yf_price)/10 # 연간 수익률의 공분산 = 일별 수익률 공분산 * 연간 평균 거래일수
#     port_risk = np.sqrt(np.dot(port_ratio.T, np.dot(annual_cov, port_ratio))) # E(Volatility) = sqrt(WT*COV*W)
#     port_risks = np.append(port_risks, port_risk)

# sorted_shape_idx = np.argsort(port_returns/port_risks)
# sorted_risk_idx = np.argsort(port_risks)


# plt.figure(figsize=(12,6))
# sns.scatterplot(x=port_risks, y=port_returns, c=port_returns/port_risks, cmap='cool', alpha=0.85, s=20)
# sns.scatterplot(x=port_risks[sorted_shape_idx[-1:]], y=port_returns[sorted_shape_idx[-1:]], color='r', marker='^', s=500)
# sns.scatterplot(x=port_risks[sorted_risk_idx[:1]], y=port_returns[sorted_risk_idx[:1]], color='b', marker='v', s=500)

# plt.title('Return per unit risk')
# plt.show()


# port_df = pd.DataFrame(port_ratios)
# sorted_port_df = port_df.iloc[sorted_shape_idx[::-1]] # 역순
# sorted_port_df.columns = yf_price.columns

# plt.figure(figsize=(12,4))
# plt.stackplot(np.arange(1,len(sorted_port_df)+1,1), np.array(sorted_port_df.T), labels=sorted_port_df.columns)

# plt.xlim(0,10000)
# plt.legend(bbox_to_anchor=(1.12,0.95))
# plt.xlabel('Ranking of Sharpe Ratio')
# plt.ylabel('Portfolio Ratio')
# plt.title('Ranking of Optimal Portfolios by Sharpe Ratio')
# plt.show()


# sorted_returns = port_returns[[sorted_port_df.index]]
# sorted_risks = port_risks[[sorted_port_df.index]]

# plt.figure(figsize=(12,4))
# plt.fill_between(x=np.arange(1,len(sorted_returns)+1,1), y1=sorted_returns.tolist(), label='return')
# plt.fill_between(x=np.arange(1,len(sorted_risks)+1,1), y1=sorted_risks.tolist(), alpha=0.3, label='risk')
# plt.xlabel('Ranking of Sharpe Ratio')
# plt.ylabel('Return & Risk')
# plt.title('Returns & Risks of Portfolio by Sharpe Ratio Ranking')
# plt.legend()
# plt.show()

# print(f'최적의 포트폴리오 비율 : \n{pd.Series(sorted_port_df.iloc[0], index=sorted_port_df.columns)}')