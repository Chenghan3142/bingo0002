import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def plot_backtest():
    # 查找历史交易记录 
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reflections_path = os.path.join(base_dir, "data", "json", "reflections.json")
    
    if not os.path.exists(reflections_path):
        print("暂无回测交易记录。请先运行 main.py 产生数据。")
        return

    with open(reflections_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    df = pd.DataFrame(records)
    if 'pnl_percent' not in df.columns:
        print("回测记录不完整，缺少盈亏字段。")
        return

    # 由于我们系统里的 pnl_percent 指的是单次交易的实际盈亏占比 (例如 1.5 表示账户资产增长了 1.5%)
    # 我们将其转换为复利权益曲线 (Equity Curve)
    df['pnl_ratio'] = df['pnl_percent'] / 100.0
    
    initial_capital = 100000.0
    equity = [initial_capital]
    for pnl in df['pnl_ratio']:
        equity.append(equity[-1] * (1 + pnl))
        
    df['equity'] = equity[1:]
    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1.0

    # 计算核心指标
    total_return = (df['equity'].iloc[-1] / initial_capital - 1) * 100
    max_drawdown = df['drawdown'].min() * 100
    
    win_rate = len(df[df['pnl_ratio'] > 0]) / len(df[df['pnl_ratio'] != 0]) * 100 if len(df[df['pnl_ratio'] != 0]) > 0 else 0
    trade_count = len(df[df['decision'] != 'HOLD'])
    
    print("\n" + "="*40)
    print("       🚀 多智能体量化回测表现报告")
    print("="*40)
    print(f"总交易次数(非HOLD): {trade_count} 笔")
    print(f"累计收益率:       {total_return:.2f}%")
    print(f"最大回撤 (MDD):   {max_drawdown:.2f}%")
    print(f"交易胜率:         {win_rate:.2f}%")
    
    if trade_count == 0:
        print("说明: 虽然由于 RAG 系统没有做多依据而没有交易，但这也是量化对冲规避风险的表现。")
        return
        
    # 可视化净值曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(equity)), equity, color='tab:red', label='Agentic RAG Portfolio Equity', linewidth=2)
    plt.title(f"Agentic System Equity Curve (Total Return: {total_return:.2f}%)")
    plt.xlabel("Trade Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    save_path = os.path.join(base_dir, "data", "backtest_result.png")
    plt.savefig(save_path)
    print(f"\n📊 净值曲线已经生成并保存至: {save_path}")

if __name__ == "__main__":
    plot_backtest()
