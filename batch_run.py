import os
import subprocess
import time
import akshare as ak
import sys

def get_sector_leaders(num_sectors=50, leaders_per_sector=12):
    """
    获取指定数量板块内的龙头股票
    """
    print(f"正在拉取所有行业板块列表(请求可能需十几秒，请耐心等待)...")
    try:
        industries = ak.stock_board_industry_name_em()
        sectors = industries['板块名称'].head(num_sectors).tolist()
    except Exception as e:
        print(f"获取板块列表受限: {e}，将使用备用常见大板块...")
        sectors = ["半导体", "酿酒行业", "医疗器械", "电池", "光伏设备", "软件开发", "汽车整车", "中药", "消费电子", "化学制药"]
        sectors = sectors * (num_sectors // len(sectors) + 1)
        sectors = sectors[:num_sectors]

    print(f"成功获取 {len(sectors)} 个板块，开始拉取各板块成分股...")
    
    all_leaders = set()
    for sector in sectors:
        try:
            cons = ak.stock_board_industry_cons_em(symbol=sector)
            # 东财接口默认排序常带有市值或资金热度，直接取前N只作为代表性龙头
            leaders = cons['代码'].head(leaders_per_sector).tolist()
            all_leaders.update(leaders)
            print(f"[{sector}] 获取完成，抽取 {len(leaders)} 只")
        except Exception as e:
            print(f"板块 [{sector}] 获取异常: {e}")
            time.sleep(1)
            continue
            
    return list(all_leaders)

import concurrent.futures

def run_single_backtest(ticker):
    print(f"\n>>>>>>>>>>>>> 启动回测标的进程: {ticker} <<<<<<<<<<<<<")
    try:
        cmd = [sys.executable, "main.py", str(ticker), "500"]
        # 为了防日志输出交叉混乱，批量模式下可适当将 stdout 重定向或者仅做核心报错抓取，在此保持原样
        process = subprocess.Popen(cmd)
        process.wait()
        print(f"✅ 标的 {ticker} 结算完成！")
    except Exception as e:
        print(f"❌ 标的 {ticker} 发生异常: {e}")

def main():
    print("====================================================================")
    print("🚀 启动批量并发回测模式：50个板块 x 10+只龙头 x 2年 (~500个交易日)")
    print("⚠️ 警告：并发会极大地挤占大语言模型的 API 并发限制 (Rate Limit)，容易报 429 错误。")
    print("====================================================================\n")
    
    tickers_to_run = get_sector_leaders(num_sectors=50, leaders_per_sector=12)
    
    print(f"\n总计获取到 {len(tickers_to_run)} 只股票准备进行回测。")
    
    # 设定并发数 (即同时跑几只股票)。如果您买的大模型 API 许可属于低级别，建议设为 1~2。
    # 若速率额度极高，可以设定为 5~10
    MAX_CONCURRENT_WORKERS = 3 
    
    print(f"\n⚡ 采用进程池进行并发调度，当前设置的并行股票数为: {MAX_CONCURRENT_WORKERS}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        # map函数会自动将所有股票派发给进程池，维持最高 N 个股票同时在跑
        try:
            executor.map(run_single_backtest, tickers_to_run)
        except KeyboardInterrupt:
            print("\n用户人工中断！停止所有并行回测。")
            executor.shutdown(wait=False, cancel_futures=True)

if __name__ == "__main__":
    main()
