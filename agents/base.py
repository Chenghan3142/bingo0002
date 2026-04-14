# Trading Agent Base System

import os
from dotenv import load_dotenv

load_dotenv()

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.scratchpad = []

    def log(self, message: str):
        print(f"[{self.name} - {self.role}] {message}")

    def query_llm(self, prompt: str) -> str:
        """调用真实的 LLM API 进行推理"""
        api_key = os.getenv("API_KEY", "your_siliconflow_or_deepseek_api_key_here")
        base_url = os.getenv("API_BASE", "https://api.deepseek.com/v1") # 默认给以DeepSeek为例的通用API兼容层
        model_name = os.getenv("MODEL_NAME", "deepseek-chat")
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": f"你是一个专业的金融量化系统中的 {self.role}。请简明扼要、客观理性地回答。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600, # 原来60会导致JSON被截断解析失败，现调大以容纳完整JSON和思维链
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.log(f"⚠️ LLM API 调用失败: {e}")
            return "[模拟降级] 基本面向好，近期发布了若干利好政策。"

    def step(self, task: str) -> str:
        raise NotImplementedError("Each agent must implement its own step logic.")
