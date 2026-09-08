---
title: “HERMES”
date: 2026-01-08 17:29:01
mathjax: true
tags:
---

## HERMES: A Unified Self-Driving World Model for  Simultaneous 3D Scene Understanding and Generation

==ICCV2025==

### 1 论文总览

#### 1.1 研究问题

自动驾驶中VLMs模型在驾驶环境的理解能力上占优势，但是缺乏对场景未来演化预测的能力；现有的DWM主要局限于场景生成任务，缺乏对***场景理解能力***的融合。具体来说就是DWMs在预测环境将如何演化方面表现出色，但是却难以对环境进行解释和描述，无法回答与场景相关的问题或提供相关的上下文信息。==对LMDrive中一系列指令以及多模态输入影响是否很大？==由此引出一个关键问题：**如何将世界知识与未来场景演化无缝集成到一个统一的世界模型中**

#### 1.2 核心思想

本文提出了一种将理解与生成任务相结合的统一世界模型HERMES。HERMES将LLM的能力进行了拓展，使其能够在自动驾驶场景中同时具备未来场景预测与大尺度空间环境理解的能力。

#### 1.3 创新点及贡献

1. 提出将理解与生成任务相结合的统一世界模型
2. 多视角图像直接转换为文本Token不仅会超出Token限制，还难以有效捕捉不同视角之间的交互关系。因此，本文提出通过BEV表示对输入进行Token化处理。
3. 本文提出使用原始BEV特征来初始化一组world queries，随后这些queries通过LLM中的因果注意力机制从文本Token中获取世界知识进行增强，最后通过这些融入世界知识的queries与经LLM处理后的BEV特征进行交互，并通过“当前-未来”实现连接，确保生成的场景演化过程同样被世界知识增强。

### 2 方法论

#### 2.1 驾驶世界模型

DWMs目的是通过预测未来场景，从大规模的无标注数据中学习到对世界的通用表征，从而使模型能够掌握真实情景下的数据分布规律。当给定$t$时刻的观测$\mathcal{O}_t$，模型将预测下一时刻的观测$\mathcal{O}_{t+1}$，其基本框架为：
$$
\mathcal{L}_t=\mathcal{E}\left(\mathcal{O}_t\right),\mathcal{L}_{t+1}=\mathcal{M}\left(\mathcal{L}_t\right),\mathcal{O}_{t+1}=\mathcal{D}\left(\mathcal{L}_{t+1}\right)
$$
其中，$\mathcal{E}和\mathcal{D}$分布表示场景的编码器和解码器，世界预测器$\mathcal{M}$则负责将隐状态$\mathcal{L}_t$映射到下一时刻的隐状态$\mathcal{L}_{t+1}$

#### 2.2 整体架构

![HERMES 整体架构](/images/hermes-overview.png)

HERMES是一个统一的驾驶世界模型，目标是同时完成场景理解（VQA）和未来场景生成（预测未来3s点云演化）。

首先输入多视角图像$I_t$，通过BEV Tokenizer编码得到BEV token——$\mathcal{F}_t$，以获取场景语义信息，并将其输入到LLM中进行处理。同时，文本token（用户问题$\mathcal{T}$）会与$\mathcal{F}_t$一起输入到LLM中用于理解当前驾驶场景，产生“世界知识”。HERMES最关键的组件为World queries，$\mathcal{F}_t$通过Max pooling提取最强的空间语义响应生成$\mathcal{Q}^w$，随后$\mathcal{Q}^w$作为后置token输入到LLM中读取世界知识进行未来的生成。最后通过Current to future link生成未来BEV，将$B_t,B_{t+1}...$进行Shared Render进行未来点云生成。

#### 2.3 World Tokenizer and Render

World Tokenizer首先对多视角图像进行处理，将其转换为压缩的连续BEV表征，随后该表征会被输入至LLM进行处理。Render模块则将BEV特征转换为点云，从而生成场景的集合信息。

##### 2.3.1 BEV-based World Tokenizer

时刻$t$的多视角图像$I_t$首先依次输入到==CLIP图像编码器==和未经过改动的==单帧BEVFormer v2==进行处理，得到BEV特征$\mathcal{F}_t^{bev} \in \mathbb{R}^{w \times h \times c}$，其同时包含语义信息和几何信息（其中$w$和$h$表示编码后场景的空间尺度，数值越大细节越丰富，$c$为BEV的通道维度）。由于其tokens体量过大，不适合直接输入LLM，因此本文设计了一个下采样模块，将$\mathcal{F}_t^{bev}$缩小两倍，得到维度为$\mathbb{R}^{\frac{w}{4} \times \frac{h}{4} \times (c \times4)}$。在被输入至LLM时，该特征会被展平并映射为$\mathcal{F}_t \in \mathbb{R}^{\frac{w}{4} \times \frac{h}{4} \times C}$。

##### 2.3.2 ==BEV-to-Point Render==

为将下采样后的BEV特征映射为场景点云$P_t$，本文首先将压缩后的BEV特征（或经过LLM处理和输出投影后的编码BEV$B_t$）通过最邻近插值与卷积操作上采样回形状$\mathbb{R}^{w \times h \times c}$。同时增加额外的高度为度来弥补BEV特征中缺乏的高度信息，将形状重塑为$\mathbb{R}^{w \times h \times z \times c_z}$。随后应用一些了3D卷积来重建体素特征$\mathcal{F}_t^{vol} \in \mathbb{R}^{w \times h \times z \times c'}$。最后根据数据集的LiDAR配置构建射线几何$\{r_k\}_{k=1}^K$，并利用可微分体渲染为每一条射线计算对应的深度。

该渲染过程将环境建模为一种隐式的符号距离函数场SDF，以精确捕获复杂而细致的几何结构信息。具体来说，给定一条起点为$o$，方向为$t_k$的射线$r_k$，将其离散化为$n$个采样点：
$$
\{p_i = o+d_it_k|i=1,...,n,0<=d_i<=d_{i+1}\}
$$
其中，$p_i$表示三维空间中的一个位置，其位置由该点在射线方向上的深度$d_i$所确定。

对于每一个采样点，通过三线形插值从体素表示$F_t^{vol}$中获取对应的局部特征嵌入$f_i$，随后利用一个浅层的MLP来预测该点的SDF值，即：
$$
s_i = \phi_{SDF}(p_i,f_i)
$$
在获得预测的SDF值后，通过对所有采样深度进行加权积分来计算渲染深度$\tilde{d}(r_k)$：
$$
\tilde{d}(r_k)=\sum^{n}_{i=1}w_id_i
$$
其中权重$w_i=T_i\alpha_{i}$表示一种无偏且具备遮挡感知能力的权重。此处透射率：
$$
T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)
$$
用于累积光子在到达第$j$个采样点之前的存活概率；不透明度
$$
\alpha_i = \max\left( \frac{\sigma_t(s_i) - \sigma_t(s_{i+1})}{\sigma_t(s_i)}, \, 0 \right)
$$
用于描述该采样点的遮挡程度。其中$\sigma_{t}(x)=(1+e^{-tx})^{-1}$是一个由可学习参数$t$调制的Simodi函数。

#### 2.3 融合

本章节主要讲解HERMES是如何将场景理解和未来场景生成融合在一起的。LLM首先基于用户指令，对来自世界分词器输出的特征$\mathcal{F}_t$进行处理，从而理解当前的驾驶场景。同时$\Delta t$组world queries从对话中汇聚世界知识，用于辅助未来场景生成。

##### 2.3.1 LLM

LLM是HERMES的核心组成部分，主要负责对BEV输入特征$\mathcal{F}_t$进行建模，解析用户指令和从真实驾驶场景的问答中获取知识并生成相应的预测结果。本文采用的LLM是==InternVL2==。

##### 2.3.2 理解

本文通过一个两层的MLP将展平后的BEV特征映射到形状为为$\mathbb{R}^{L_{bev} \times C}$的表示，并将其输入到LLM的特征空间中。其中$L_{bev}$表示输入BEV的长度，$C$为LLM的通道维度。对于针对当前场景的提示语，本文将其分词为不同的词表索引和文本Token$\mathcal{T}$，并输入至LLM进行处理。本文模型能够对用户关于驾驶环境的提问作出响应，生成场景描述并回答视觉问答问题，**通过下一token预测的方式来理解场景。**

##### 2.3.3 生成

为了使LLM具备未来场景生成的能力，本文提出了**world queries**，用于将世界知识与未来场景关联起来，并增强LLM与渲染模块之间的信息传递。

首先，文章对tokenizer中得到的flattened BEV特征$\mathcal{F}_t \in \mathbb{R}^{\frac{w}{4}\times\frac{h}{4}\times C}$进行最大池化，从其峰值响应中提取world queries，得到$\mathcal{Q} \in \mathbb{R}^{n \times (c \times 4)}$。随后，将$\mathcal{Q}$复制$\Delta{t}$次，形成queries组$\{\mathcal{Q}_i|i=1,...,\Delta{t}\}$。同时为了进一步实现可控的未来生成，本文将==自车运动条件编码==为$e_{t+i}$，该向量用于描述自车从当前时刻到第$i$帧的规划位置与航向，并映射为高维嵌入表示。随后，将自车运动信息$e_{t+i}$加入到对应的queries$\mathcal{Q_i}$中。此外，本文还引入了==帧嵌入==$FE \in \mathbb{R}^{\Delta{t}\times(c \times 4)}$，并通过广播机制加入，用于指示每一组world queries所负责的预测时间帧。最后，由于$\mathcal{Q}_w$与$\mathcal{F}_t$共享一个语言空间投影层（MLP），因此投影后的$\mathcal{Q}_w \in \mathbb{R}^{(\Delta{t}\times n)\times C}$。总结如下：
$$
\mathcal{Q}_w=MLP(Concat[\mathcal{Q}_i + e_{t+i}|i=\{1,...,\Delta{t}\}]+FE)
$$
==LLM的因果注意力机制==（即后续token可以访问先前的信息）使得world queries能够获取在场景理解过程中获得的世界知识。经过LLM前向传播处理后，编码后的BEV特征与world queries会通过一个共享的两层MLP从LLM的通道维度$C$投影回$c \times 4$。

由于每一组world queries仅包含n个查询，这只能提供对未来世界的稀疏表示，从而给world render重建未来场景带来困难。因此本文提出了*current-to-future link*模块，该模块通过==交叉注意力层==，将世界知识注入到未来BEV特征中。

*current-to-future link*包含三个交叉注意力块，用于生产未来的BEV特征。每个交叉注意力块由以下部分组成：

- 一个交叉注意力层，其中使用来自 LLM 输出的编码 BEV 特征 $B_t$ 作为 Query，而每个场景对应的 world queries 作为 Key 和 Value；
- 随后的一个自注意力层和一个前馈网络，用于进一步处理空间信息。

最终，编码后的 BEV 特征 $B_t$ 以及生成的未来 BEV 特征 $(B_{t+1}, \cdots, B_{t+\Delta t})$ 被送入一个共享的世界渲染模块（world Render），从而得到从当前时刻 $P_t$ 到未来时刻 $P_{t+\Delta t}$ 的点云预测结果。
