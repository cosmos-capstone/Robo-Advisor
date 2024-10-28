import FinanceDataReader as fdr
import pandas as pd

# 모든 한국 주식 종목 데이터를 불러오기
krx = fdr.StockListing('KRX')
stock_data_list = []

# 모든 종목의 데이터를 수집하여 리스트에 저장
for code, name in zip(krx['Code'], krx['Name']):
    try:
        data = fdr.DataReader(code, '2024-9-25', '2024-10-28')
        data['Code'] = code  # 종목 코드 추가
        data['Name'] = name  # 종목 이름 추가
        stock_data_list.append(data)
        print(stock_data_list)
    except:
        pass  # 데이터가 없는 종목은 건너뜁니다.

# 모든 종목 데이터 합치기
stock_data = pd.concat(stock_data_list)
print(stock_data)