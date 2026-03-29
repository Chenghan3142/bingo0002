import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class StockTrendPredictor(nn.Module):
    """
    简单的 LSTM 模型用于预测带有连续因子的时间序列趋势
    """
    def __init__(self, input_size=10, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        predictions = self.linear(last_out)
        return predictions[0]

class DLEngine:
    def __init__(self, weight_path="models/weights/pretrained_model.pth"):
        self.model = StockTrendPredictor(input_size=10, hidden_layer_size=64, output_size=1)
        
        # 解析绝对路径
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.weight_path = os.path.join(base_dir, weight_path) if not os.path.isabs(weight_path) else weight_path
        
        self.scaler = StandardScaler()
        self._load_or_init_weights()

    def _load_or_init_weights(self):
        import os
        if os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path))
            print(f"[DL Engine] 深度学习预测模型权重 [{self.weight_path}] 加载成功.")
        else:
            print("[DL Engine] 未发现预训练权重，随机初始化...")
            torch.save(self.model.state_dict(), self.weight_path)

    def train_on_history(self, df_hist: pd.DataFrame, window_size=10, epochs=20, lr=0.005):
        """真实使用前置历史数据训练网络"""
        print("[DL Engine] 启动真实数据环境训练...")
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        if len(df_hist) <= window_size:
            print("[DL Engine] 数据不足，跳过训练。")
            return

        # 准备数据集
        raw_data = df_hist[cols].values
        self.scaler.fit(raw_data)
        scaled_data = self.scaler.transform(raw_data)

        X, y = [], []
        for i in range(len(scaled_data) - window_size - 1):
            X.append(scaled_data[i : i + window_size])
            # 预测第二天涨跌幅作为标签
            target_pnl = df_hist.iloc[i + window_size + 1]['涨跌幅'] / 10.0 # 缩放标签以便收敛
            y.append([target_pnl])
            
        if not X: return

        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.FloatTensor(np.array(y))

        for epoch in range(epochs):
            optimizer.zero_grad()
            # 简化为不使用 batch 直接过一轮全量数据(仅为快速跑通)
            preds = []
            for i in range(len(X_tensor)):
                out = self.model(X_tensor[i:i+1])
                preds.append(out)
            
            preds_tensor = torch.stack(preds)
            loss = criterion(preds_tensor, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 5 == 0:
                print(f"  -> Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

        # 持久化
        torch.save(self.model.state_dict(), self.weight_path)
        print(f"[DL Engine] 模型使用真实数据微调完毕，已保存至 {self.weight_path}")

    def predict(self, ticker: str, features: np.ndarray) -> dict:
        self.model.eval()
        with torch.no_grad():
            if not hasattr(self.scaler, 'mean_'):
                self.scaler.fit(features[-10:])
            scaled_features = self.scaler.transform(features[-10:])
            seq = torch.FloatTensor(scaled_features).unsqueeze(0)
            raw_score = self.model(seq).item()

        confidence = min(np.abs(raw_score) * 100, 99.9)
        trend = "上涨 (看多)" if raw_score > 0 else "下跌 (看空)"
        return {
            "score": round(raw_score * 10, 4), # 放大因为预测时压了一倍
            "trend": trend,
            "confidence": f"{round(confidence, 2)}%"
        }
