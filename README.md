
# 基于多模态超声动态影像的乳腺肿瘤精确诊断方法

## 项目简介

本项目旨在通过多模态超声动态影像（CEUS和B-mode）结合深度学习方法，开发一套乳腺肿瘤精确诊断系统。该系统能够通过实时的动态超声影像，提供精准的肿瘤分类和分割结果，辅助医学影像分析，提高乳腺肿瘤的诊断效率与准确性。

## 项目结构

```
BME_UI/
├── ceus_dual_task/              # 超声影像处理与模型代码
│   ├── dual_tasks_code/         # 主要任务相关代码
│   │   ├── __pycache__/         # 缓存文件
│   │   ├── segmentation/        # 分割模型文件
│   │   ├── segmentation_result/ # 分割结果
│   │   ├── data.py              # 数据加载与处理
│   │   ├── frozen_resnet50_lstm_model.py # 训练冻结的ResNet50-LSTM模型
│   │   ├── main.py              # 主程序
│   │   ├── model.py             # 模型定义
│   │   ├── refer_ceus.py        # 参考CEUS图像处理
│   │   ├── refer.py             # 参考代码
│   │   ├── test.py              # 测试代码
│   │   ├── train.py             # 训练代码
│   ├── model_weights/           # 模型权重文件
│   ├── for_test/                # 测试相关文件
├── static/                      # 静态文件
│   ├── image/                   # 图片资源
│   ├── info/                    # 信息
│   ├── output/                  # 输出文件
│   ├── analytic.js              # 分析脚本
│   ├── home.js                  # 主页脚本
│   ├── README.md                # 项目说明文件
│   ├── README.pdf               # 项目说明PDF文件
│   ├── script.js                # 脚本文件
│   ├── start.css                # 页面样式
│   ├── start.js                 # 页面交互脚本
│   ├── style.css                # 样式文件
│   ├── style2.css               # 辅助样式
│   ├── style3.css               # 辅助样式
├── templates/                   # HTML模板
│   ├── home.html                # 主页模板
│   ├── main.html                # 主界面模板
│   ├── start.html               # 启动页面模板
├── app.py                       # Flask应用主程序
```

## 功能模块

1. **图像分类与分割模型**：
   - 使用ResNet50作为编码器模块，提取B-mode和CEUS影像的空间特征。
   - 使用Unet作为解码器，进行影像的分割任务。
   - 基于LSTM提取时间特征，进行影像分类。
   - 双任务协同训练：同时进行影像分类和分割任务。
   
2. **数据处理**：
   - 数据集包含来自中大医院的乳腺肿瘤影像数据（包括B-mode和CEUS图像），每个样本包含60帧CEUS图像、一张B-mode图像和一个分割mask。
   - 数据集按4:1:1的比例划分为训练集、验证集和测试集。
   - 预处理包括调整图像尺寸为224x224，并进行归一化处理。

3. **训练与评估**：
   - 使用五折交叉验证来评估模型的性能。
   - 分类任务评估指标包括准确率、精确率、召回率、F1分数和混淆矩阵。
   - 分割任务评估指标包括DICE系数、像素准确率和交并比。

4. **界面设计**：
   - 提供基于Flask框架的用户界面（UI），可以进行诊断结果展示、历史记录查询等功能。

## 安装与使用

### 环境要求

- Python 3.6+
- 必要的Python依赖：`torch`, `torchvision`, `flask`, `opencv-python`, `scikit-learn`等。

### 安装步骤

1. 克隆本项目：
   ```bash
   git clone <项目仓库地址>
   cd BME_UI
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 运行Flask应用：
   ```bash
   python app.py
   ```

4. 访问项目主页：
   打开浏览器，访问 `http://127.0.0.1:5000/`。

## 项目进展

目前，项目已完成以下阶段：

1. 数据集的收集与预处理；
2. 模型的初步训练与评估；
3. 基于B-mode和CEUS影像的双任务模型设计与实现。

## 未来计划

1. 继续改进B-mode分支网络模型，提升诊断准确率；
2. 探索更先进的CEUS与B-mode图像融合方法，如引入注意力机制；
3. 深入研究基于时空分析的模型优化，提升分类和分割精度；
4. 在实际医疗环境中进行模型验证与部署。

