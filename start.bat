@echo off
chcp 65001 >nul
echo ==================================================
echo   TradingAgents_RAG_DL 环境一键配置脚本 (Windows)
echo ==================================================

:: 1. 检查并创建虚拟环境
IF NOT EXIST "venv" (
    echo [*] 未检测到虚拟环境，正在为您创建基于 Python 的 venv 虚拟环境...
    python -m venv venv
    echo [+] 虚拟环境创建成功！
) ELSE (
    echo [*] 虚拟环境已存在。
)

:: 2. 激活虚拟环境
echo [*] 正在激活虚拟环境...
call venv\Scripts\activate.bat

:: 3. 安装依赖
echo [*] 正在安装所需的依赖包 (使用清华镜像源加速)...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 4. 检查 .env 配置文件
IF NOT EXIST ".env" (
    echo [*] 未检测到 .env 配置文件，正在从 .env.example 复制默认配置...
    copy .env.example .env >nul
    echo [!] 警告: 已生成默认 .env 文件。请打开 .env 文件并填入您的大模型 API_KEY！
    echo [*] 填入后请重新运行此脚本启动项目。
    pause
    exit /b
)

:: 5. 启动主程序
echo [*] 环境检查完毕，正在启动系统...
python main.py
pause
