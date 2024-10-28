import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import datetime

class FDRData():
    def __init__(self, tickers, period):
        self.period = period

        self.df_price = pd.DataFrame()
        for ticker in tickers:
            print(f'데이터 불러오는 중 (ticker : {ticker})')
            full_data = fdr.DataReader(ticker, start=self.period[0], end=self.period[1])
            self.df_price[ticker] = full_data['Close']  
            time.sleep(0.5) 
        print(f'데이터셋 준비 완료')
            
    def get_price(self):
        return self.df_price
    
    
def get_year(start_date, end_date):
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
    year_diff = end_date.year - start_date_obj.year
    return year_diff