---
title: "AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive  Reasoning and Reinforcement Fine-Tuning"
date: 2026-09-17 18:33:01
mathjax: true
tags:
  - 论文阅读
  - VLA推理增强
categories:
  - 技术
---



# AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive  Reasoning and Reinforcement Fine-Tuning

## 1 Overview of the paper

### 1.1 Research Questions 

1. **VLM 的语义推理空间与车辆的连续物理动作空间之间存在接口问题**。直接让语言模型输出数值 waypoint 容易出现不合理轨迹；加一个独立 planner 或 decoder，又会让系统复杂化，削弱统一端到端训练的意义。
2. **VLM 推理计算量没有按场景难度分配**。简单直路也生成长 CoT 很浪费，复杂交互场景直接输出 action 又可能缺乏推理。

### 1.2 Core Concepts

本文通过**码本映射**连接了动作空间与VLM的推理空间，同时将推理token与动作token融合实现联合优化，再利用SFT和RFT实现了快慢双通道推理。

### 1.3 Innovation and Contribution

1. 采样K-disk从真实驾驶数据中采样了2048个典型action token，并将动作空间中的连续轨迹切片后与典型action token一一对应，将原本的连续轨迹转换为适合VLM处理的离散token。
2. 将reasoning与action统一在一个Transformer中，没有专门设计一个trajectory planner(ORION)，减少了模型冗余。
3. SFT+RFT实现了双通道快慢推理。

## 2 Methodology

![image-20260812100513336](/images/autovla-overview.png)

AutoVLA框架由两个主要部分构成：

1. VLM Backbone：该模块处理视觉输入和文本输入，并生成相应的token，包括推理token和动作token。
2. 物理动作Token生成：本文拓展了语言模型的解码器，使其能够输出与车辆运动直接对应的物理动作token。



AutoVLA训练分为两个阶段：

1. 监督微调：利用真实轨迹数据进行训练，同时从大规模VLM中蒸馏高质量的推理数据。
2. 强化微调：利用针对具体任务设计的奖励函数来优化规划性能，同时通过减少不必要的推理过程，提高模型的运行效率。

![image-20260812102247608](/images/autovla-training.png)

### 2.1 Framework

#### 2.1.1 Model Inputs

1. 多视角、多帧相机数据$C$：车辆前方、左前方和右前方的三个RGB摄像头，其中$c_i = [c_{i}^{t-3},c_{i}^{t-2},c_{i}^{t-1},c_i^t]$。
2. 高层导航指令$I$。
3. 自车状态$S$：车辆当前的速度、加速度以及历史动作信息。

#### 2.1.2 Base VLM Model

采用Q wen2.5-VL-3B作为AutoVLA的视觉语言主干网络。

#### 2.1.3 Action Tokenization

原始轨迹是连续变量:
$$
P \in \mathbb{R}^{\tau \times d}
$$
但是Qwen2.5VL工作在有限离散词表上，如果让其直接生成连续数值轨迹模型需要通过文本token间接表达连续几何量，导致其性能下降。因此，本文目标是构建一个**连续车辆运动->有限action token**的映射，这样子轨迹规划就变成标准的自回归预测。

本文首先需要构建一个轨迹映射码本，其具体过程是先在真实轨迹中采样，但由于真实轨迹中包含大量冗余的重复动作，因此作者通过 K disk选择出2048个代表性动作，覆盖99.42%的驾驶场景，最后得到本文码本。

在训练过程中，假设训练集中有一条真实的5s轨迹，该轨迹会被分成10个0.5s的segment。对于每一个真实的segment，在码本中寻找最接近的action，随后将其映射为对应action token。至此，一条连续轨迹就变成一段离散化的token，并输入到模型中进行学习。
$$
p(a_1,a_2,\dots,a_{10} \mid \text{images},I,S)
$$
在推理阶段，模型可能生成：

```
<action_71>
<action_923>
<action_1102>
<action_783>
...
```

查 codebook：
$$
a_{71} \rightarrow (\Delta x_1,\Delta y_1,\Delta\theta_1) \\ a_{923} \rightarrow (\Delta x_2,\Delta y_2,\Delta\theta_2)
$$
依次从初始 ego pose 开始进行坐标变换和累积。

如果当前 pose 是：
$$
(x_t,y_t,\theta_t)
$$
局部 action 为：
$$
 (\Delta x,\Delta y,\Delta\theta)
$$
概念上下一 pose 会通过局部到全局坐标变换得到：
$$
x_{t+1} = x_t+ \cos\theta_t\Delta x - \sin\theta_t\Delta y\\
 y_{t+1} = y_t+ \sin\theta_t\Delta x + \cos\theta_t\Delta y\\
 \theta_{t+1} = \theta_t+\Delta\theta
$$

#### 2.1.4 Unified Reasoning and Action

AutoVLA 在单一的自回归 Transformer 框架中统一了推理过程与动作生成过程，使模型能够根据不同的驾驶场景，在快速思考与慢速思考之间进行自适应切换。

1. 快速思考：AutoVLA不生成较长的思维链推理，而是直接预测物理动作 token，从而能够在较为简单、直接的驾驶场景中实现快速响应。
2. 慢速思考：模型首先分析当前环境，识别其中的关键要素，并对潜在的后续结果进行推理，然后再决定最终的驾驶动作。

### 2.2 Reasoning Data

本文基于Qwen2.5-VL-72B模型进行自动化推理标注，该流程生成的结构化推理标注主要包含四个关键部分：

1. 详细的场景描述
2. 关键目标识别
3. 周围交通参与者意图预测
4. 合适驾驶动作的确定

### 2.3 Supervised Fine-tuning

AutoVLA的完整输出：
$$
x =  [l_1,\cdots,l_L,a_1,\cdots,a_T]
$$
第一部分$l_1,\cdots,l_T$是语言token，也就是reasoning；第二部分$a_1,\cdots,a_T$是动作token。

> **为什么SFT数据有两种？**
>
> 1. Fast Thinking
>
>    简单场景不需要推理，作者希望模型直接得到：
>
>    ```
>    无需复杂推理
>    <action_12>
>    <action_15>
>    <action_18>
>    ...
>    ```
>
>    此时$l$是一个固定的短模板，用来告诉模型不需要详细CoT。
>    $$
>    (C,I,S) \to l_{short}+[a_1,\cdots,a_T]
>    $$
>    模型训练目标：
>    $$
>    p_{\theta}(l_{short}+[a_1,\cdots,a_T]|C,I,S)
>    $$
>
> 2. Slow Thinking
>
>    复杂场景需要先分析。例如：前方施工车辆占道，右侧有车，左侧车道空闲，导航要求直行。标准答案可能是：
>
>    ```
>    Scene Analysis:
>    前方道路被施工车辆部分占据。
>          
>    Critical Objects:
>    施工车辆是主要障碍物。
>    左侧车道当前空闲。
>          
>    Intention Reasoning:
>    继续直行可能存在碰撞风险。
>    左侧换道更安全。
>          
>    Final Action:
>    向左变道并适当加速。
>          
>    <action_527>
>    <action_631>
>    <action_812>
>    ...
>    ```
>
>    对应：
>    $$
>    (C,I,S) \to [l_1,\cdots,l_L]+[a_1,\cdots,a_T]
>    $$
>    模型训练目标：
>    $$
>    p_\theta (CoT+action tokens | C,I,S)
>    $$



作者使用了两个损失函数
$$
\mathcal L_{\mathrm{LM}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
p_\theta
(x_i\mid x_{<i},C,I,S)
\\
\mathcal L_{\mathrm{action}}
=
-\frac{1}{T}
\sum_{i=L+1}^{L+T}
\log
p_\theta(x_i\mid x_{<i},C,I,S)
$$
***理由：完整输出中reasoning token可能非常多，而action token少很多，那么优化器可能更关注CoT而忽略了动作。***

最后最终损失为：
$$
\mathcal L_{\mathrm{SFT}}^i
=
w_i
\left(
\mathcal L_{\mathrm{LM},i}
+
\lambda_a
\mathcal L_{\mathrm{action},i}
\right),where:
w_i=
\begin{cases}
\lambda_{\mathrm{cot}}, & \text{有 CoT}\\
1, & \text{没有 CoT}
\end{cases}
$$

### 2.4 Reinforcement Fine-tuning

AutoVLA在SFT完成后已经有了一个模型：$\pi_{SFT}$，该模型已经学会从场景得到reasoning tokens和action tokens，RFT的目标是在这个基础上继续优化轨迹质量和推理效率。

首先从训练数据中抽取一个scenario U构造输入：$q = (C,I,S)$，模型会针对同一个$q$采样$G$个不同输出：
$$
o = \{o_1,o_2,\cdots,o_G\}
$$
每个$o_i$中都有action tokens，通过前面的码本将其解码成$\tau_i$，用来后面评价轨迹质量。

作者定义：
$$
r
=
r_{\mathrm{Driving}}
-
\lambda_r r_{\mathrm{CoT}}
$$
可以理解为总分=驾驶质量-啰嗦程度惩罚。

对于nuPlan/NAVSIM，作者直接用PDMS代表$r_{Driving}$；对于Waymo，作者定义：
$$
r_{\mathrm{Driving}}
=
\frac{
\delta-ADE
}{
\kappa
}
\\
where：ADE
=
\frac{1}{T}
\sum_{t=1}^{T}
\|
\hat y_t-y_t
\|_2
$$
对于CoT penalty，作者定义：
$$
r_{\mathrm{CoT}}
=
\frac{
1
}{
1+
e^{-(L-L_{\mathrm{tol}})\gamma}
}
$$
其中，$L$表示CoT token长度，$L_tol$表示容忍长度，$\gamma$表示惩罚曲线陡峭程度。

随后进行组内归一化：$A_i = \frac{ r_i-\bar r }{ \sigma_r }$，利用$A_i$进行优化，其目标函数定义为：
$$
J_{\mathrm{GRPO}}(\theta)
=
\mathbb E_{q,\{o_i\}\sim
\pi_{\theta_{\mathrm{old}}}(O|q)}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\left(
J_R^i
-
\beta
D_{\mathrm{KL}}
(\pi_\theta\Vert\pi_{\mathrm{ref}})
\right)
\right]
\\
where:J_R^i
=
\min
\left(
\frac{\pi_\theta(o_i|q)}
{\pi_{\theta_{\mathrm{old}}}(o_i|q)}
A_i,
\operatorname{clip}
\left(
\frac{\pi_\theta(o_i|q)}
{\pi_{\theta_{\mathrm{old}}}(o_i|q)},
1-\epsilon,
1+\epsilon
\right)
A_i
\right)
$$


## 3 Experimental Design

| 模块     | 核心配置                                                     |
| -------- | ------------------------------------------------------------ |
| 数据     | nuPlan + Waymo + nuScenes + CARLA                            |
| 评测     | NAVSIM、nuScenes、Waymo E2E、Bench2Drive                     |
| Planning | 0.5 s / token，10 tokens，5 s horizon                        |
| SFT      | lr $1e^{-5}$，5 epochs，8 × L40S，batch 32，$\lambda_a=1$，$\lambda_{\mathrm{cot}}=40$ |
| RFT      | LoRA，lr $3e^{-5}$，$\beta=0.04$，6,000 steps            |
| 评测形式 | Open-loop + Closed-loop                                      |

## 4 Literature Review

### 4.1 Advantages

1. 将驾驶理解、推理和规划放入同一个autoregressive VLA模型
2. SFT双思考模式让模型同时具备fast thinking和slow thinking
3. 通过RFT的reward平衡驾驶性能和推理成本

### 4.2 Limitations

计算资源需求过高

## 5 Personal Summary

本文采用码本映射的方式联通数值轨迹空间与VLM推理空间，同时通过SFT和RFT两种训练方式实现了快慢双通道推理，大大提高了推理速度。