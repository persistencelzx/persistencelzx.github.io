---
title: ORION: A Holistic End-to-End Autonomous Driving Framework by  Vision-Language Instructed Action Generation
date: 2026-09-09 16:28:27
mathjax: true
tags:
  - 论文阅读
  - VLA推理增强
categories:
  - 技术
---

## 1 Overview of the paper

### 1.1 Research Questions 

**VLM的能力位于语义推理空间，自动驾驶规划输出位于连续数值动作空间，两者之间存在表示鸿沟。**在预训练VLM驱动的E2E自动驾驶中，如何将压缩后的视觉时序表征和语言上下文映射到连续、多模态的轨迹分布中？

### 1.2 Core Concepts

本文最大的创新点在于通过一个以VAE为基础的生成式规划器对齐推理空间与动作空间，用于指导真实轨迹生成。

### 1.3 Innovation and Contribution

1. 引入QT-Former，提高了模型聚合长期历史信息的能力
2. 引入一个生成式规划器，对齐推理空间与动作空间

## 2 Methodology

![image-20260728005304266](/Users/persistencelzx/Library/Application Support/typora-user-images/image-20260728005304266.png)

ORION首先通过一个Vison Encoder对多视角图像进行编码，将编码后的token输入到QT-Former中，QT-Former利用其三种queries（Perception,Scene,History）来聚合长期视觉上下文和交通信息。随后将聚合后的token与语言指令token通过LLM结合（**推理空间**）生成一个规划token，再将该规划token输入到规划生成器中用于连接**推理空间和动作空间**，生成未来轨迹。

### 2.1 QT-Former

![image-20260729104330417](/Users/persistencelzx/Library/Application Support/typora-user-images/image-20260729104330417.png)

本文引入了QT-Former用于处理多视角图像信息和融合历史上下文。QT-Former首先初始化三个Queries：Perception Queries（$Q_p \in \mathbb{R}^{N_p \times C_q }$）、Scene Queries（$Q_s \in \mathbb{R}^{N_s \times C_q}$）和History Queries（$Q_h\in \mathbb{R}^{N_h \times C_q}$ ）,其中$N_p,N_s,N_h$为token数，$C_q$为通道数。随后感知Queries和场景Queries在拼接后进行Self-Attention，用于交换信息，减少各自独立工作之间的信息割裂，完成后再通过索引将二者拆分开来。接着各自与经过Vision Encoder与3D位置编码处理后的图像特征进行Cross-Attention提取场景信息，其中的感知Queries被用于目标检测、交通状态识别等任务。

QT-Former通过一个长期记忆库来存储历史信息，其$M \in \mathbb{R}^{(N_h \times n)\times C_q}$。历史Queries首先与加入时间戳后的$M$进行Cross-Attention用于提取历史信息，之后再与当前场景Queries进行Cross-Attention读取当前场景信息（通过历史信息可以知道当前场景应重点关注的地方）。随后按照先进先出规则更新历史记忆库，最后将场景Queries和历史Queries送入到MLP中映射为LLM的推理空间。
$$
I_t
\xrightarrow{\text{Vision Encoder}}
F_m^t
\\


[Q_p,Q_s]
\xrightarrow{\text{Self-Attention}}
[Q_p',Q_s']

\\
[Q_p',Q_s']
\xrightarrow{\text{Cross-Attention with }F_m^t}
[\widetilde{Q}_p^t,\widetilde{Q}_s^t]
\\
Where:\widetilde{Q}_p^t
\longrightarrow
\begin{cases}
\text{目标检测},\\
\text{交通状态},\\
\text{运动预测}.
\end{cases}
\\
Q_h
\xrightarrow{\text{读取 Memory Bank}}
Q_h'
\\
Q_h'
\xrightarrow{\text{查询当前 }Q_s^t}
\widehat{Q}_h^t
\\
\widehat{Q}_h^t
\longrightarrow
M_t
\\
Q_s^t,\widehat{Q}_h^t
\xrightarrow{\mathrm{MLP}}
x_s,x_h
\\

[x_s,x_h,x_q]
\longrightarrow
\mathrm{LLM}
\longrightarrow
\text{planning token}
\\


\text{planning token}
\longrightarrow
\text{Generative Planner}
\longrightarrow
\text{未来轨迹}
$$

### 2.2 LLM

场景tokens$x_s$、历史tokens$x_h$和用户指令tokens$x_q$被一起输入到大模型中，与此同时，作者为 LLM 设计了一个规划问答模板，并在最后一个问答任务中引入特殊的规划 token \(s\)，用于将整个驾驶场景的理解与推理上下文汇聚到该 token 中。其形式化表示为：
$$
s \sim p(s \mid x_s, x_h, x_q, x_a)
$$
其中，\(x_a\) 表示 LLM 生成的回答。规划 token \(s\) 的嵌入表示将作为条件，用于控制后续的轨迹生成。

### 2.3 Generative Planner

训练时有两个输入：

1. Planning tokens

   输入$s$，经过一个MLP encoder得到\(p(z_s|s) = N(\mu_s,\sigma_s^2)\)，表示这个驾驶语义对应的动作潜变量分布。

2. 专家轨迹

   训练时提供一个专家轨迹$t$用于指导Planning tokens，经过MLP得到\(p(z_t|t) = N(\mu_t,\sigma_t^2)\)，表示这条真实驾驶轨迹对应的动作潜变量分布。

ORION 假设：

如果 LLM 理解正确，那么：\[ \text{语义} \approx \text{对应动作} \]

因此：\[ p(z_s|s) \approx p(z_t|t) \]，使用 KL 散度：\[ \mathcal{L}_{vae} = D_{KL} ( p(z_s|s) || p(z_t|t) ) \]

得到动作 latent：\[ z \]之后使用 GRU decoder，论文采用 GenAD 中的 GRU decoder。

> 流程：
> $$
> z \rightarrow GRU \rightarrow \hat{\tau}
> $$
> 输出：
> $$
> \hat{\tau} = \{ (\hat{x}_1,\hat{y}_1), ... (\hat{x}_T,\hat{y}_T) 
> $$
> 即未来轨迹。

## 3 Experimental Design

1. 数据集：Bench2drive
2. 评估指标
   - 闭环：Driving Score (DS), Success Rate (SR), Efficiency, Comfortness, Multi-Ability.
   - 开环：L2 distance error 和 the collision rate
   - VQA：CIDEr , BLEU, 和 ROUGE-L
3. 视觉编码器：EVA-02-L
4. LLM：Vicuna v1.5

## 4 Literature Review

### 4.1 Advantages

1. 提出 reasoning-action 对齐框架；
2. QT-Former 建模长期历史；
3. VQA 多任务增强理解；
4. Bench2Drive 闭环性能显著提升

### 4.2 Limitations

1. 因果推理缺少直接证据；
2. 性能提升归因困难；
3. VAE 对齐理论不足；
4. 泛化和部署验证不足

### 4.3 Proposed Improvements 

1. 增加反事实因果验证；
2. 模型轻量化；
3. 加强真实道路测试；
4. 引入世界模型/RL；
5. 提升 planning token 可解释性

## 5 Personal Summary

本文核心是引入一个基于VAE的生成式规划器，该规划器弥补了推理空间与动作空间之间的鸿沟。但是其资源消耗过大，后续应寻找更加轻量化的模型。
