import akshare as ak

print("Getting industries...")
industries = ak.stock_board_industry_name_em()
print(industries.head())

industry_name = industries['板块名称'].iloc[0]
print(f"Getting constituents for {industry_name}...")
cons = ak.stock_board_industry_cons_em(symbol=industry_name)
print(cons.head())
