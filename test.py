import FinanceDataReader as fdr

print(fdr.DataReader('148070'))

from pykrx import bond
 
df = bond.get_otc_treasury_yields('20240325')
print(df)