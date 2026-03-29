from .base import BaseDataProvider
import akshare as ak

class AkShareProvider(BaseDataProvider):
    def fetch_sentiment_data(self, ticker: str):
        try:
            news_df = ak.stock_news_em(symbol=ticker)
            if not news_df.empty:
                return " | ".join(news_df['新闻标题'].head(3).tolist())
            return "近期无明显新闻热点。"
        except Exception as e:
            return f"实时新闻数据查询受限: {e}"

    def fetch_fundamental_data(self, ticker: str):
        try:
            info_df = ak.stock_individual_info_em(symbol=ticker)
            info_dict = dict(zip(info_df['item'], info_df['value']))
            industry = info_dict.get('行业', '未知')
            market_cap = info_dict.get('总市值', '未知')
            
            # 由于东方财富基础接口不再包含动态市盈率和市净率，这里使用百度估值接口获取最新的财务估值数据
            try:
                pe_df = ak.stock_zh_valuation_baidu(symbol=ticker, indicator='市盈率(TTM)', period='近一年')
                pe = pe_df['value'].iloc[-1] if not pe_df.empty else '未知'
                
                pb_df = ak.stock_zh_valuation_baidu(symbol=ticker, indicator='市净率', period='近一年')
                pb = pb_df['value'].iloc[-1] if not pb_df.empty else '未知'
            except Exception:
                pe, pb = '未知', '未知'

            return f"所属行业: {industry}, 总市值: {market_cap}, 动态市盈率(PE): {pe}, 市净率(PB): {pb}"
        except Exception as e:
            return f"财务API请求遇到阻碍: {e}"

    def fetch_macro_data(self):
        try:
            sh_df = ak.stock_zh_index_daily(symbol="sh000001")
            recent_sh = sh_df.tail(3)[['date', 'close', 'volume']].to_dict('records')
            return f"上证指数最近3个交易日表现: {recent_sh}"
        except Exception as e:
            return f"上证大盘数据抓取报错: {e}"

    def fetch_smart_money_data(self, ticker: str):
        try:
            market = "sh" if str(ticker).startswith("6") else "sz"
            fund_df = ak.stock_individual_fund_flow(stock=ticker, market=market)
            recent_fund = fund_df.tail(2)[['收盘价', '主力净流入-净额', '涨跌幅']].to_dict('records')
            return f"近2个交易日的大单主力资金净流入(元)与价格变动特征: {recent_fund}"
        except Exception as e:
            return f"主力资金API抓取异常: {e}"
