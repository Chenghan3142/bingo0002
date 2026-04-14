import streamlit as st
import subprocess
import json
import pandas as pd
import os

st.set_page_config(page_title="多智能体量化推演系统", layout="wide", page_icon="📈")

# --- Custom CSS ---
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stButton>button {
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        font-size: 16px;
    }
    .st-bw {
        border-radius: 10px;
    }
    .metric-value {
        font-size: 32px !important;
        font-weight: bold !important;
    }
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2F3C7E;
        font-weight: 800;
    }
    h2, h3 {
        color: #3f51b5;
    }
    
    /* 自定义精美日志的滚动容器 */
    .log-container {
        background-color: #1e1e2e;
        color: #cdd6f4;
        padding: 20px;
        border-radius: 10px;
        font-family: 'Courier New', Courier, monospace;
        height: 500px;
        overflow-y: auto;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    .log-line { margin: 4px 0; line-height: 1.5; font-size: 14px; }
    .log-date { color: #f38ba8; font-weight: bold; font-size: 16px; margin-top: 15px; margin-bottom: 5px;}
    .log-sys { color: #a6adc8; font-style: italic; }
    .log-analyst { color: #89b4fa; }
    .log-rag { color: #cba6f7; font-weight: bold; }
    .log-debate { color: #fab387; }
    .log-referee { color: #f9e2af; font-weight: bold; }
    .log-risk { color: #a6e3a1; font-weight: bold; }
    .log-success { color: #a6e3a1; }
    .log-normal { color: #bac2de; }
</style>
""", unsafe_allow_html=True)

st.title("📈 A股智能投研多智能体系统")
st.markdown("##### 基于 **Agentic RAG**、**深度学习 (LSTM)** 与 **多智能体博弈架构** 的量化回测与实盘推演平台。")
st.markdown("---")

# 侧边栏
st.sidebar.header("⚙️ 推演参数配置")
ticker = st.sidebar.text_input("股票代码", value="000001", help="输入A股代码，如000001或600519")
days = st.sidebar.number_input("推演天数", min_value=1, max_value=750, value=30, help="向前推演运行的交易日天数")

if st.sidebar.button("▶️ 启动深度推演", type="primary"):
    st.toast(f"正在对 {ticker} 展开 {days} 天的多智能体兵棋推演...", icon="🔥")
    
    # 建立一个占位符用来流式刷新精美日志
    log_container = st.expander("🕵️ 实时多智能体兵棋推演沙盘", expanded=True)
    log_placeholder = log_container.empty()
    
    with st.spinner("🚀 系统总指挥部已下达调令，智囊团正在进行数据挖掘、多空辩论与量化演算..."):
        process = subprocess.Popen(
            ["python", "-u", "main.py", str(ticker), str(days)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ, HTTP_PROXY="", HTTPS_PROXY="", http_proxy="", https_proxy="", ALL_PROXY="", all_proxy="") # 绕过系统代理防止爬虫被阻断
        )
        
        logs = []
        for line in process.stdout:
            logs.append(line.strip())
            if len(logs) > 50:  # 缓存最新的 50 行以渲染 UI
                logs.pop(0)
            
            # 美化日志行
            formatted_html = "<div class='log-container'>"
            for log in logs:
                html_class = "log-normal"
                icon = "⚙️"
                
                # 正则/关键词匹配改变颜色和图标
                if "时间游标滑动" in log: 
                    html_class = "log-date"
                    icon = "📅"
                    log = f"<hr style='border:1px dashed #4e4e6a; margin:10px 0;'> {log}"
                elif "System" in log: 
                    html_class = "log-sys"
                    icon = "💻"
                elif "分析师" in log:
                    html_class = "log-analyst"
                    icon = "📊"
                elif "特工" in log or "RAG" in log:
                    html_class = "log-rag"
                    icon = "🔎"
                elif "辩手" in log:
                    html_class = "log-debate"
                    icon = "🤺"
                elif "裁判" in log:
                    html_class = "log-referee"
                    icon = "⚖️"
                elif "风控" in log or "交易接口" in log:
                    html_class = "log-risk"
                    icon = "🛡️"
                elif "✅" in log or "💎" in log or "全新底层原则" in log:
                    html_class = "log-success"
                    icon = "✨"

                formatted_html += f"<div class='log-line {html_class}'><span>{icon}</span> {log}</div>"
            formatted_html += "</div>"
            
            log_placeholder.markdown(formatted_html, unsafe_allow_html=True)
            
        process.wait()
    
    st.success("🏁 各作战兵团推演杀青！战场与情报数据已全部沉淀入知识库。")

# --- 战报可视化 ---
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
reflections_path = os.path.join(data_dir, "json", "reflections.json")
principles_path = os.path.join(data_dir, "json", "principles.json")

st.markdown("---")
st.subheader("📊 智能投研数据中心")

# 创建 Tab 页签来分离内容
tab_overview, tab_history, tab_principles = st.tabs(["🚀 策略总览 (Dashboard)", "📝 详细交易日志", "💎 知识图谱与核心公理"])

with tab_overview:
    if os.path.exists(reflections_path):
        try:
            with open(reflections_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            
            if records:
                df = pd.DataFrame(records)
                if 'pnl_percent' in df.columns:
                    df['pnl_ratio'] = df['pnl_percent'] / 100.0
                    initial_capital = 100000.0
                    equity = [initial_capital]
                    for pnl in df['pnl_ratio']:
                        equity.append(equity[-1] * (1 + pnl))
                    
                    df['equity'] = equity[1:]
                    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1.0

                    # 计算指标
                    total_return = (df['equity'].iloc[-1] / initial_capital - 1) * 100
                    max_drawdown = df['drawdown'].min() * 100
                    valid_trades = df[df['pnl_ratio'] != 0]
                    win_rate = len(df[df['pnl_ratio'] > 0]) / len(valid_trades) * 100 if len(valid_trades) > 0 else 0
                    trade_count = len(df[df['decision'] != 'HOLD'])
                    
                    # 使用卡片化布局顶部指标
                    st.markdown("### 🏆 核心绩效指标 (Key Performance Indicators)")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    with st.container():
                        m1.metric(label="📈 累计复利收益率", value=f"{total_return:.2f}%", delta=f"{total_return:.2f}%")
                        m2.metric(label="📉 最大回撤 (MDD)", value=f"{max_drawdown:.2f}%", delta=f"{max_drawdown:.2f}%", delta_color="inverse")
                        m3.metric(label="🎯 产生交易胜率", value=f"{win_rate:.2f}%")
                        m4.metric(label="🔄 总策略捕捉单数", value=f"{trade_count} 笔")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📈 策略复利净值曲线 (Equity Curve)")
                    st.area_chart(df['equity'], use_container_width=True, color="#4CAF50")
            else:
                st.info("💡 暂无有效交易数据，请先点击侧边栏运行一次推演。")
        except Exception as e:
            st.error(f"读取记录日志失败, error: {e}")
    else:
        st.warning("⚠️ 暂无日志文件 (reflections.json)，请执行推演系统初始化数据。")

with tab_history:
    st.markdown("### 🔍 详细历史交易日志矩阵")
    if os.path.exists(reflections_path):
        try:
            with open(reflections_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if records:
                st.dataframe(pd.DataFrame(records), use_container_width=True, height=600)
            else:
                st.info("数据为空。")
        except:
            st.error("解析文件失败。")
    else:
        st.warning("无历史日志。")

with tab_principles:
    st.markdown("### 🧠 Agent 底层公理引擎")
    st.markdown("这里集中展示从过去的错误和获利经历中提取出的、由 **自我迭代大模型结晶出的高维交易原则**。通过向量空间持久化指导未来的决策。")
    if os.path.exists(principles_path):
        try:
            with open(principles_path, "r", encoding="utf-8") as f:
                pr = json.load(f)
            if pr:
                st.dataframe(pd.DataFrame(pr), use_container_width=True, height=500)
            else:
                st.info("知识库暂为空原则数据。")
        except:
            st.error("解析历史反思准则发生错误。")
    else:
        st.warning("无知识库。")
