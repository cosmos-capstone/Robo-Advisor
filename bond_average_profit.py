from pykrx import bond
from datetime import date
import pandas as pd

# 국고채 수익률 평균 계산 및 복리 이자 계산 함수
def get_average_treasury_yield(n):
    all_yields = []
    current_year = date.today().year  
    base_date = '1006'   # 월,일은 어케하지?

    for i in range(n):
        year = str(current_year - i)  
        date_str = year + base_date  

        df = bond.get_otc_treasury_yields(date_str)

        bonds = ['국고채 1년', '국고채 2년', '국고채 3년', '국고채 5년']
        bond_yields = df.loc[bonds, '수익률']

        average_yield = bond_yields.mean()
        all_yields.append(average_yield)
        avg = sum(all_yields) / len(all_yields)
    return avg


if __name__ == "__main__":
    n = 4 
    avg = get_average_treasury_yield(n)
    print(avg)
