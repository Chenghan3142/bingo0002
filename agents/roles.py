from .base import BaseAgent
import numpy as np
import time
from dataflows.providers.akshare_provider import AkShareProvider

provider = AkShareProvider()

# ==========================================
# 1. 数据与基础分析师团队 (Analysts)
# ==========================================

class TechnicalAnalyst(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "技术面分析师")

    def step(self, ticker: str, features: np.ndarray, target_date: str = None):
        self.log(f"分析 [{ticker}] 的K线形态、均线与动量指标...")
        data_str = "暂无足够技术面数据"
        if features is not None and len(features) > 0:
            last_few_days = features[-5:] if len(features) > 5 else features
            data_str = f"过去几天的数据特征张量 (例如标准化后的开盘、收盘等): {np.round(last_few_days, 4).tolist()}"

        self.log(f"✅ 获取到技术面输入数据: {data_str[:150]}...")
        prompt = f"你是一个专业的技术面量化分析师。针对股票 {ticker}，以下是它最新截面的量价技术面数据：\n{data_str}\n请判断技术面当前呈现出的涨跌倾向。请在回答最后明确包含 'positive', 'negative'，或 'neutral' 中的一个英文单词代表情绪。你的理由尽量简短(50字以内)。"
        llm_result = self.query_llm(prompt)
        self.log(f"LLM根据技术面数据推理得出: {llm_result}")
        
        sentiment = "neutral"
        if "positive" in llm_result.lower(): sentiment = "positive"
        elif "negative" in llm_result.lower(): sentiment = "negative"
        
        return {"agent": self.name, "sentiment": sentiment, "reasoning": llm_result}

class SentimentAnalyst(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "舆情分析师")

    def step(self, ticker: str, target_date: str = None):
        self.log(f"挖掘 [{ticker}] 社交媒体(股吧、雪球等)新闻散户情绪...")
        news_str = provider.fetch_sentiment_data(ticker)
        self.log(f"✅ 成功抓取舆情: {news_str}")

        prompt = f"你是一名专门对接对冲基金的散户舆情与新闻情感分析师。对于股票代码 {ticker}，相关市场舆情如下：\n{news_str}\n请你研判整体散户与新闻面传递出的情绪是多头还是空头。请在回答段落最后明确输出 'positive', 'negative'，或 'neutral'。附带简短推理逻辑(50字以内)。"
        llm_result = self.query_llm(prompt)
        self.log(f"LLM根据舆情数据推理得出: {llm_result}")
        
        sentiment = "neutral"
        if "positive" in llm_result.lower(): sentiment = "positive"
        elif "negative" in llm_result.lower(): sentiment = "negative"
        
        return {"agent": self.name, "sentiment": sentiment, "reasoning": llm_result}

class FundamentalAnalyst(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "基本面分析师")

    def step(self, ticker: str, target_date: str = None):
        self.log(f"分析 [{ticker}] 财报数据(PE, PB)及行业研报估值...")
        data_str = provider.fetch_fundamental_data(ticker)
        self.log(f"✅ 成功抓取基本面核心估值数据: {data_str}")

        prompt = f"你是一名资深的价值投资基本面分析师。针对股票 {ticker}，以下是最新的真实基本面估值数据：\n{data_str}\n结合该行业的普遍情况与估值分布（如PE/PB是否具备安全边际），研判当前基本面健康度。请在结尾明确包含 'positive', 'negative' 或 'neutral'。 给出极简分析逻辑(50字内)。"
        llm_result = self.query_llm(prompt)
        self.log(f"LLM根据基本面数据推理得出: {llm_result}")
        
        sentiment = "neutral"
        if "positive" in llm_result.lower(): sentiment = "positive"
        elif "negative" in llm_result.lower(): sentiment = "negative"
            
        return {"agent": self.name, "sentiment": sentiment, "reasoning": llm_result}

class MacroAnalyst(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "宏观经济分析师")

    def step(self, ticker: str, target_date: str = None):
        self.log(f"评估宏观周期、利率环境及大盘(上证指数)系统性风险...")
        macro_str = provider.fetch_macro_data()
        self.log(f"✅ 成功抓取近期上证大盘走势: {macro_str[:50]}...")

        prompt = f"你是一名宏观经济及大盘系统性风险分析师。结合以下A股上证大盘近期的真实指标：\n{macro_str}\n请判断当前市场整体系统性环境、流动性情绪对做多个股是否具备支撑。回答结尾必须明确输出 'positive', 'negative' 或 'neutral'，理由需非常精简(不超过50字) 。"
        llm_result = self.query_llm(prompt)
        self.log(f"LLM根据宏观大盘数据推理得出: {llm_result}")
        
        sentiment = "neutral"
        if "positive" in llm_result.lower(): sentiment = "positive"
        elif "negative" in llm_result.lower(): sentiment = "negative"
            
        return {"agent": self.name, "sentiment": sentiment, "reasoning": llm_result}

class SmartMoneyAnalyst(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "主力资金分析师")

    def step(self, ticker: str, target_date: str = None):
        self.log(f"监控 [{ticker}] 北向资金、机构龙虎榜与大单净流入...")
        flow_str = provider.fetch_smart_money_data(ticker)
        self.log(f"✅ 成功提取主力大单资金净流入: {flow_str[:50]}...")

        prompt = f"你是量化团队中的主力游资（Smart Money）追踪监测分析师。对于股票 {ticker}，以下是你捕获到的最新主力净流入异动数据：\n{flow_str}\n请判断游资或机构主力目前是在洗盘吸筹、出货派发还是观望。最后一行必须包含 'positive', 'negative' 或 'neutral'。 理由请控制在50字左右。"
        llm_result = self.query_llm(prompt)
        self.log(f"LLM根据游资流入数据推理得出: {llm_result}")
        
        sentiment = "neutral"
        if "positive" in llm_result.lower(): sentiment = "positive"
        elif "negative" in llm_result.lower(): sentiment = "negative"
            
        return {"agent": self.name, "sentiment": sentiment, "reasoning": llm_result}

class NewsAnalystAgent(BaseAgent):
    def __init__(self, name: str, rag_engine):
        super().__init__(name, "新闻研报分析师")
        self.rag = rag_engine

    def step(self, ticker: str, target_date: str = None):
        date_str = f"({target_date})" if target_date else ""
        self.log(f"根据 RAG 知识库检索新闻并推演情感...")
        docs = self.rag.retrieve(ticker, target_date=target_date, top_k=2)
        
        if not docs:
            docs = ["暂无有效新闻资讯"]
            
        summary = "综合文献得到：" + ", ".join(docs)
        self.scratchpad.append(f"RAG 检索完成 - {summary}")
        
        # 为了加速本地回测，使用本地规则短路，如果是实盘可放开 query_llm
        if "利好" in summary or "大增" in summary:
            llm_conclusion = "基本面显著向好，具备上涨潜力(positive)"
            sentiment = "positive"
        else:
            # 依然保留一段查询以展示真实的 API 接入，若无 KEY 会退化为模拟
            analysis_prompt = f"针对新闻：【{summary}】，给出极简涨跌判定(positive/negative)。"
            llm_conclusion = self.query_llm(analysis_prompt)
            sentiment = "positive" if "positive" in llm_conclusion.lower() or "涨" in llm_conclusion else "negative"
            
        self.log(f"大模型研判结论: {llm_conclusion}")
        return {"agent": self.name, "sentiment": sentiment, "reasoning": llm_conclusion}

class QuantResearcherAgent(BaseAgent):
    def __init__(self, name: str, dl_engine):
        super().__init__(name, "深度学习量化研究员")
        self.dl = dl_engine

    def step(self, ticker: str, features_override: np.ndarray = None, target_date: str = None):
        self.log(f"输入张量特征执行 LSTM 深度模型推演...")
        if features_override is None:
            features = np.random.rand(10, 10)
        else:
            features = features_override

        pred = self.dl.predict(ticker, features)
        sentiment = "positive" if pred["score"] > 0 else "negative"
        self.log(f"LSTM 预测: {pred['trend']} 置信度[{pred['confidence']}]")
        return {"agent": self.name, "sentiment": sentiment, "reasoning": f"DL逻辑倾向 {pred['trend']} (打分:{pred['score']:.2f})"}


# ==========================================
# 2. 辩论与博弈层 (Debate & Game Theory)
# ==========================================

class BullResearcher(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "看多逻辑辩手")

    def step(self, reports: list):
        bull_points = [r for r in reports if r['sentiment'] == 'positive']
        self.log(f"提炼看多阵营子弹: 共收集 {len(bull_points)} 条看多逻辑。")
        
        if not bull_points:
            return {"agent": self.name, "strength": 0, "thesis": "暂无有效的看多逻辑支撑。"}
            
        info_str = "\n".join([f"- {r['agent']}: {r['reasoning']}" for r in bull_points])
        prompt = f"你目前代表【多方阵营】。这里是各分析师给出的看多信号：\n{info_str}\n请将它们凝聚成一篇精炼有力、具有说服力的多头辩论陈词(字数在100字以内)。"
        thesis = self.query_llm(prompt)
        self.log(f"看多辩词: {thesis}")
        return {"agent": self.name, "strength": len(bull_points), "thesis": thesis}

class BearResearcher(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "看空逻辑辩手")

    def step(self, reports: list):
        bear_points = [r for r in reports if r['sentiment'] == 'negative']
        self.log(f"提炼看空阵营子弹: 共收集 {len(bear_points)} 条看空逻辑。")
        
        if not bear_points:
            return {"agent": self.name, "strength": 0, "thesis": "暂无有效的看空逻辑支撑。"}
            
        info_str = "\n".join([f"- {r['agent']}: {r['reasoning']}" for r in bear_points])
        prompt = f"你目前代表【空方阵营】。这里是各分析师给出的看空或利空风险信号：\n{info_str}\n请将它们凝聚成一篇精炼有力、具有警示性的空头辩论陈词(字数在100字以内)。"
        thesis = self.query_llm(prompt)
        self.log(f"看空辩词: {thesis}")
        return {"agent": self.name, "strength": len(bear_points), "thesis": thesis}

class GameReferee(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "多空博弈裁判")

    def step(self, bull_case, bear_case):
        self.log("⚖️ 正在进行高维裁判...")
        
        bull_thesis = bull_case["thesis"]
        bear_thesis = bear_case["thesis"]
        bull_score = bull_case["strength"]
        bear_score = bear_case["strength"]
        
        prompt = f"""你是一名极为理性的量化交易主理人(裁判)。请仔细聆听以下多空双方的辩论陈词，以及对应的子阵营分析师票数厚度：

【多方阵营陈词】 (票数: {bull_score})
{bull_thesis}

【空方阵营陈词】 (票数: {bear_score})
{bear_thesis}

现请你综合裁断，最终只能给出：'BUY'、'SELL' 或 'HOLD' 作为操作结论。
你的回答格式必须是：
结论：<BUY/SELL/HOLD>
理由：<不多于100字的裁判裁决原因>
"""
        llm_result = self.query_llm(prompt)
        
        decision = "HOLD"
        if "结论：BUY" in llm_result or "结论: BUY" in llm_result or "结论： BUY" in llm_result:
            decision = "BUY"
        elif "结论：SELL" in llm_result or "结论: SELL" in llm_result or "结论： SELL" in llm_result:
            decision = "SELL"
            
        self.log(f"裁判决断 -> 【{decision}】 深度理由:\n{llm_result}")
        return {"decision": decision, "reason": llm_result}


# ==========================================
# 3. 执行与风控层 (Execution & Risk)
# ==========================================

class RiskManager(BaseAgent):
    def __init__(self, name: str, memory_bank):
        super().__init__(name, "首席风控官")
        self.memory_bank = memory_bank

    def step(self, ticker: str, referee_decision: dict):
        final_decision = referee_decision["decision"]
        reason = referee_decision["reason"]
        
        self.log("🛡️ 进行交易前的最后风控审查...")
        if self.memory_bank:
            # 【进阶方向1&5】：从高维向量数据库中调取高度相似的历史风险场景（取代生硬的时间倒推）
            try:
                scene_desc = f"当前计划对 {ticker} 执行 {final_decision}，裁判理由: {reason}"
                similar_experiences = self.memory_bank.retrieve_relevant_experience(scene_desc, role="System", current_regime="General", top_k=2)
                
                if similar_experiences:
                    self.log(f"🧠 联想到了 {len(similar_experiences)} 条相关的历史深刻教训...")
                    # 假如有极低分（亏损严重）的高相似度经验警告了该操作：
                    for exp in similar_experiences:
                        if "重创" in exp['content'] or "严重回撤" in exp['content']:
                            if final_decision != "HOLD":
                                final_decision = "HOLD"
                                reason += f" | 向量风控阻断: 提取到惨痛历史类似教训警示: {exp['content'][:50]}..."
                                self.log("⚠️ 触发[高维向量相似度检索]熔断，否决当前高危提议！")
                                break
            except AttributeError:
                pass
                
            past_memories = self.memory_bank.get_recent_reflections(k=3, ticker=ticker)
            if past_memories:
                has_large_drawdown = any(mem.get('pnl_percent', 0) < -2.0 for mem in past_memories if mem.get('decision') in ["BUY", "SELL"])
                if has_large_drawdown and final_decision == "BUY":
                    final_decision = "HOLD"
                    reason += " | 基础风控阻断: 历史近期存在 >2% 重大回撤，强制熔断转为 HOLD！"
                    self.log("⚠️ 触发近期基础记忆风控熔断，否决买入提议！")
                else:
                    self.log("✅ 基础风控审核通过，近期无重大回撤隐患。")
        return {"decision": final_decision, "reason": reason}

class TraderAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "交易执行机器人")

    def step(self, final_instruction: dict) -> str:
        decision = final_instruction["decision"]
        self.log(f"💰 接收到上游最终指令: {decision}。 向券商柜台/撮合引擎发送订单。")
        return decision

class QuantitativeRiskReflector(BaseAgent):
    def __init__(self, name: str, memory_bank):
        super().__init__(name, "量化分析与策略反思官")
        self.memory_bank = memory_bank
        
    def step(self, ticker: str, decision: str, reports: list, pnl_percent: float):
        self.log(f"T+1日模拟结算复盘，标的[{ticker}]真实单次盈亏 [{pnl_percent}%]")
        
        all_records = [m for m in self.memory_bank.memory if m.get('ticker') == ticker and m.get('decision') in ["BUY", "SELL"]]
        current_trade_pnl = pnl_percent if decision in ["BUY", "SELL"] else 0.0
        
        historical_pnls = [m.get('pnl_percent', 0.0) for m in all_records]
        if decision in ["BUY", "SELL"]:
            historical_pnls.append(current_trade_pnl)
            
        N = len(historical_pnls)
        if N > 0:
            win_pnls = [p for p in historical_pnls if p > 0]
            loss_pnls = [p for p in historical_pnls if p < 0]
            
            p_win = len(win_pnls) / N
            avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
            avg_loss = abs(sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0.0
            pnl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            if pnl_ratio != float('inf') and pnl_ratio > 0:
                kelly_fraction = p_win - ((1 - p_win) / pnl_ratio)
            else:
                kelly_fraction = p_win
                
            kelly_fraction = max(0.0, min(1.0, kelly_fraction)) * 100
            expected_return = p_win * avg_win - (1 - p_win) * avg_loss
            
            stats_msg = f"样本={N}笔, 胜率={p_win*100:.1f}%, 盈亏比={pnl_ratio if pnl_ratio != float('inf') else 999.99:.2f}, 期望单次收益={expected_return:.2f}%. Kelly建议仓位: {kelly_fraction:.1f}%"
        else:
            stats_msg = "暂无足够实盘买卖样本计算凯利仓位与胜率。"

        reflection_text = ""
        if decision in ["BUY", "SELL"]:
            if current_trade_pnl < -3.0:
                reflection_text = f"严重回撤 ({current_trade_pnl}%)！当前统计: {stats_msg}。系统应考虑降低贝塔参与度。"
            elif current_trade_pnl > 0:
                reflection_text = f"策略获利 ({current_trade_pnl}%)。当前统计: {stats_msg}。"
            else:
                reflection_text = f"微小摩擦 ({current_trade_pnl}%)。当前统计: {stats_msg}"
        else:
            reflection_text = "本次为空仓观望(HOLD)，未产生资金损患。"
            
        record = {
            "ticker": ticker,
            "decision": decision,
            "pnl_percent": round(pnl_percent, 2),
            "reflection_text": reflection_text,
            "math_stats": stats_msg
        }
        
        # 将新经验记录，并且触发RAG后台自动向量化并打标签
        self.memory_bank.append(record)
        
        # 【进阶方向4】：经验反馈形成强化学习闭环
        # 如果有真实的 PnL 盈亏出现，根据盈亏扣除或者增加过去相同标的、相同决策下累积相似经验的权重！
        if decision in ["BUY", "SELL"] and pnl_percent != 0:
            self.memory_bank.update_experience_score_by_action(ticker, decision, pnl_percent)
            
        # 【进阶方向3】：偶尔尝试触发结晶
        try:
            high_value_materials = self.memory_bank.crystallize_knowledge(None)
            if high_value_materials:
                self.log(f"🧠 [自我进化] 检测到了高分致胜经验池，正在呼叫大模型将其结晶为公理法则！")
                prompt = f"你是一名为对冲基金撰写《内部交易原则》(Redbook)的总监。以下是系统在实盘后积累的、被验证赚了钱的优质高分经验片段：\n{high_value_materials}\n请你将这些碎片提炼为1条【高度抽象、具有普适性的量化交易公理】，不能超过40个字，要求极度精炼。"
                principle_text = self.query_llm(prompt)
                
                # 存入到原则文件中，并更新向量库标记（略去复杂标记以防覆盖，只存json）
                import datetime
                self.memory_bank.principles.append({
                    "date": datetime.datetime.now().strftime('%Y-%m-%d'),
                    "ticker": ticker,
                    "principle": principle_text
                })
                self.memory_bank._save_principles()
                self.log(f"📜 [全新底层原则已确立并入库]: {principle_text}")
        except Exception as e:
            pass

        self.log(f"【进化总结】: {reflection_text}")
        return reflection_text
