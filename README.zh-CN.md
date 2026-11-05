# 微信跳一跳自动辅助 — macOS

这是一个基于 **Python、OpenCV、NumPy 和 macOS Quartz** 的个人工程项目，能够从微信小游戏窗口截图中识别棋子与下一平台，计算长按时长，并在确认画面稳定、上一跳落地成功后继续跳跃。

**它使用传统计算机视觉，不是训练后的 AI 神经网络。** 当目标不唯一、证据不足、画面过期、窗口被遮挡或落点无法确认时，会安全停止，而不是盲目跳跃。项目非微信 / 腾讯官方产品。

[English README](README.md)

![程序实际生成的视觉识别结果：通过 OpenCV 分析游戏画面，提取平台轮廓、识别下一跳目标，并结合几何投影计算预计落点。图中的辅助线与标记用于展示自动决策过程。

](assets/detection-demo.png)

## 运行环境

- Mac + 微信桌面客户端及「跳一跳」小游戏。
- 建议使用 Python 3.11 或更新版本。
- 在「系统设置 → 隐私与安全性」为启动程序所用的终端授予**屏幕录制**及**辅助功能**权限；不同 macOS 版本名称略有区别。
- 游戏窗口保持完整、无遮挡，运行中不要调整尺寸。

## 安装

在解压或 clone 的项目目录里运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp settings.example.json settings.json
```

`settings.json` 是本机配置，不会上传 GitHub。里面的系数 1700 和角度 30° 来自特定环境的标定，不保证所有电脑都通用。

## 使用

```bash
python jump.py --diagnose          # 检查权限及游戏窗口，不发送鼠标输入
python jump.py --auto --dry-run    # 只做自动识别预览，不实际跳跃
python jump.py --auto             # 全自动连续跳跃
python jump.py --auto --max-jumps 3  # 最多三跳
python jump.py --loop             # 半自动：每跳自己选择目标
```

启动自动模式后，在倒计时内将微信游戏窗口置于前台，把鼠标移到**游戏内容区域的空白处**并松开。程序使用原生 Quartz 鼠标事件发送长按。

**停止方式：** 按 Esc、移动鼠标或切换窗口；识别歧义、无法确认落地等情况也会自动停止。程序不会自动重开游戏。

## 项目结构

- `vision.py`：按优先级组织识别流程。
- `vision_pixels.py` / `vision_geometry.py` / `vision_validation.py`：公共颜色、几何及候选验证。
- `vision_edges.py` / `vision_materials.py` / `vision_small.py` / `vision_occlusion.py`：平台边缘、材质、小平台和遮挡识别。
- `vision_tracking.py` / `vision_landing.py`：跨帧追踪及落地确认。
- `aiming.py` / `mac_input.py`：固定角度投影与 macOS 鼠标控制。
- `test_*.py` / `testdata/`：测试代码与可复现的游戏图像案例。

## 测试

```bash
python -m unittest discover -s . -p 'test_*.py'
```

完整测试需要 macOS / Quartz 环境。部分历史回放数据包含本地运行记录，没有包含在公开仓库中。项目曾在实际运行中实现过连续 100 多跳，但不保证每个关卡或每台设备都成功。

本项目仅供工程学习和个人实验，请遵守相关平台规则。`testdata/` 中的游戏截图保留原游戏美术素材的第三方权利；本仓库未授予开源许可证，源代码与原创文档保留相关权利。
