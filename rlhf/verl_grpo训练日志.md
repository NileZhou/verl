##  基础版本
1. /njfs/train-nlp/huzheng/train_agentrl/agentrl1119_grpo001.sh


遇到问题
1. KL 0.003→0.006 是“假上升”：绝对值仍远低于 0.05，说明策略还没真正偏离初始点；此时不是 KL 限制了更新，而是优势信号本身太小或为负，导致 pg_loss 正负随机漂。
2. 优势几乎为 0 → 检查 reward 基线/归一化
3. pg_clipfrac 仅 2 % → 更新量被 clip 吃掉了；说明要么 ε 设得太小（默认 0.2，可临时提到 0.3），要么优势信号太小（回到第 2 点）。
4. reward不抬头
1. 目前“纹丝不动”的核心瓶颈是 学习率太小 + 优势信号被压平 + rollout 方差不够




## 优化参数版本
1. /njfs/train-nlp/huzheng/train_agentrl/agentrl1119_grpo002.sh

主要优化方向
1. 提高temperature 、rollout 数量、学习率


## 优化版本
1. /njfs/train-nlp/huzheng/train_agentrl/agentrl1119_grpo003.sh

主要优化方向：
1. 跑起来14b


## 优化版本
1. /njfs/train-nlp/huzheng/train_agentrl/agentrl1119_grpo003.sh

主要优化方向：
1. 优化奖励模型，在80～100步里面，发现会进行格式探索：["think", "plan", "web_search", "observation", "think", "answer"] -> ["think", "plan", "think", "web_search", "observation", "think", "answer"]；所以扩充了推理链路的格式形式。
2. 
        






# 参考指标


| 指标                           | 物理意义                                | 健康区间 / 趋势                                                                 | 看到异常时怎么拧                                                           |
| ---------------------------- | ----------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **pg\_loss**                 | 策略在“优势”指引下的改进量，负值越大→更新越猛            | 训练前 50 步从 -0.1 → -0.01 快速回升，后期保持在 -0.02～0.005 小幅波动；**绝对值持续反弹**＝lr 太大或优势爆炸 | 1. 先查 advantages/std 是否 >0.3；<br>2. 若 std 够仍反弹 → lr 减半             |
| **kl\_loss** (KL(πθ∥πref))   | 当前策略离初始 SFT 的远近，防止“跑飞”              | 缓慢**单调上升**，终点 ≤ 0.05（0.1 是红线）；**陡增** → 马上降 lr 或加大 β                       | 1. kl > 0.03 时 β×1.5；<br>2. kl > 0.08 时 lr×0.5                     |
| **pg\_clipfrac**             | 被 PPO-clip 截断的 token 比例，反映“实际更新幅度”  | 10 %–30 %；**<5 %**＝优势没拉开；**>40 %**＝ε 太小或优势爆炸                              | <5 % → 先做 reward whitening 或放大 reward 2–5×；>40 % → ε 0.2→0.1 或降 lr |
| **advantages/mean**          | 平均优势，告诉模型“整体做对还是做错”                 | 应**从 0 开始持续抬升到 0.2 以上**；若一直 −0.1~0.1 抖动＝没学会                               | 先打 reward/std、value/std；reward std <0.1 → 放大奖励或换奖励函数               |
| **advantages/std**           | 优势信号的“区分度”，直接决定 clipfrac 和收敛速度      | **≥ 0.3** 才算合格；<0.05 时训练基本停滞                                              | 同 advantages/mean，先做 whitening 或放大奖励                               |
| **entropy**                  | 策略输出的熵，探索性指标                        | 起点 0.6–0.8，**平滑下降到 0.25–0.35** 后稳住；**断崖式下跌** → 死记                         | entropy <0.25 时把 entropy\_coeff 从 0.01 提到 0.03                     |
| **reward/mean**（验证集）         | 最终效果的“唯一真神”                         | 应**单调抬升**，每 50 步至少 +2 %；平台期 >100 步 → 调数据/奖励                               | 若 reward 震荡，先把 lr 降 30 % 再观察                                       |
| **value/mean**               | Critic 网络预测的期望回报，用来算优势              | 趋势跟 reward/mean 同步即可；**严重偏离** → Critic 初始化/学习率问题                          | value 与 reward 差 >0.5 → Critic lr 降 2×                             |
| **grad\_norm**               | 全局梯度范数，早爆探测器                        | 平稳区间 0.5–2.0；**突然 >5.0** 伴随 pg\_loss 反弹 → 梯度爆炸                            | 立即 lr×0.5，或开 grad-clip=1.0                                         |
| **clipfrac(v)**（Critic clip） | Critic 更新被 clip 的比例，辅助看 value 网络稳定性 | 5 %–20 %；**>30 %** → value 网络步长太大                                         | 把 critic\_lr 降到 actor\_lr 的 1/2