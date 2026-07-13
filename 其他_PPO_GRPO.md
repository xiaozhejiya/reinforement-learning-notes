# PPO 与 GRPO 算法讲义

> 适用对象：已经了解基础深度学习，希望理解大模型 RLHF / RLVR 中 PPO 与 GRPO 关系、公式和训练流程的学习者。  
> 核心结论：**GRPO 不是完全脱离 PPO 的新范式，而是 PPO-Clip 在大模型强化学习场景下的 critic-free 改造版。**

---

## 目录

1. PPO 与 GRPO 的直接关系  
2. PPO 基础符号：状态、动作、奖励、回报、价值与优势  
3. PPO 的核心：概率比值与 clip 操作  
4. PPO 的 value function 与 return target  
5. PPO 在大语言模型中的含义  
6. GRPO 的核心思想：组内相对优势  
7. GRPO 中的随机采样、$R_i$ 与 $A_i$
8. PPO 与 GRPO 的完整对比  
9. GRPO 的主要不足  
10. 当前比原始 GRPO 更值得关注的替代或改进方法  
11. DeepSeek 系列与 GRPO 的关系  
12. 一页速记版  
13. 参考资料  

---

# 1. PPO 与 GRPO 的直接关系

PPO，全称 **Proximal Policy Optimization**，是经典的 on-policy policy gradient 算法。

GRPO，全称 **Group Relative Policy Optimization**，是 DeepSeekMath / DeepSeek-R1 系列中使用的大模型强化学习算法。

二者最直接的关系是：

```text
GRPO ≈ PPO-Clip + Group-relative advantage - Value/Critic model
```

换句话说：

- PPO 通过 **critic / value model** 估计 advantage；
- GRPO 去掉 critic / value model；
- GRPO 使用同一个 prompt 下多个随机采样回答的**组内相对 reward** 来估计 advantage；
- GRPO 仍然保留 PPO 的核心：**新旧策略概率比值 + clip 限制 + KL 约束**。

所以，GRPO 不是 DPO 那类离线偏好优化方法，也不是完全脱离 PPO 的新算法。它仍然属于 policy gradient / on-policy RL 思路。

---

# 2. PPO 基础符号

在强化学习里，常见符号如下。

| 符号 | 含义 |
|---|---|
| $s_t$ | 第 $t$ 步的状态 |
| $a_t$ | 第 $t$ 步采取的动作 |
| $r_t$ | 第 $t$ 步得到的即时奖励 |
| $\pi_\theta(a_t \mid s_t)$ | 策略模型，在状态 $s_t$ 下选择动作 $a_t$ 的概率 |
| $V_\phi(s_t)$ | value function，对状态 $s_t$ 未来累计回报的估计 |
| $\hat R_t$ | 从状态 $s_t$ 开始的累计回报目标 |
| $\hat A_t$ | advantage，表示动作相对于平均预期好多少或差多少 |
| $\gamma$ | 折扣因子 |
| $\epsilon$ | PPO clip 范围超参数 |

---

## 2.1 $V_\phi(s_t)$ 是什么？

理论上的 value function 定义为：

$$
V^\pi(s_t)
=
\mathbb{E}_\pi
\left[
\sum_{k=t}^{T}
\gamma^{k-t} r_k
\mid
s_t
\right]
$$
意思是：

从状态 $s_t$ 出发，如果后面一直按照策略 $\pi$ 行动，未来能获得的折扣累计奖励期望是多少。

神经网络版本就是：

$$
V_\phi(s_t)
\approx
V^\pi(s_t)
$$

它输出的是一个标量分数，不是概率。

---

## 2.2 $\hat R_t$ 是什么？

$$
\hat R_t
$$
不是“获得 $r_t$ 的概率”。

它是从 $s_t$ 开始往后实际采样到的或估计出来的**累计回报目标**。

最简单的 Monte Carlo 写法是：

$$
\hat R_t
=
r_t
+
\gamma r_{t+1}
+
\gamma^2 r_{t+2}
+
\cdots
+
\gamma^{T-t}r_T
$$

也就是：

$$
\hat R_t
=
\sum_{k=t}^{T}
\gamma^{k-t} r_k
$$
例如：

$$
r_t=0,\quad r_{t+1}=0,\quad r_{t+2}=1,\quad \gamma=0.9
$$

那么：

$$
\hat R_t
=
0 + 0.9\times 0 + 0.9^2\times 1
=
0.81
$$
这个 $0.81$ 是累计回报，不是概率。

---

## 2.3 advantage 是什么？

最简单写法：

$$
\hat A_t
=
\hat R_t - V_\phi(s_t)
$$

它表示：

```text
实际回报 - 预期回报
```

如果：

$$
\hat A_t > 0
$$
说明实际结果比 critic 预期更好，这个动作应该被鼓励。

如果：

$$
\hat A_t < 0
$$

说明实际结果比 critic 预期更差，这个动作应该被抑制。

---

# 3. PPO 的核心：概率比值与 clip 操作

PPO 定义新旧策略的概率比：

$$
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t \mid s_t)
}{
\pi_{\theta_{\text{old}}}(a_t \mid s_t)
}
$$
它表示：

```text
新策略生成这个动作的概率，是旧策略的多少倍。
```

例如：

- $\rho_t=1.2$：新策略生成该动作的概率变成旧策略的 1.2 倍；
- $\rho_t=0.7$：新策略生成该动作的概率下降到旧策略的 0.7 倍；
- $\rho_t=1.0$：新旧策略对该动作的概率相同。

---

## 3.1 `clip()` 是什么？

`clip()` 就是截断 / 限幅 / clamp 操作。

数学定义：

$$
\operatorname{clip}(x,a,b)
=
\begin{cases}
a, & x<a \\
x, & a\le x\le b \\
b, & x>b
\end{cases}
$$

也可以写成：

$$
\operatorname{clip}(x,a,b)=\min(\max(x,a),b)
$$
在 PPO 中：

$$
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)
$$

表示把 $\rho_t(\theta)$ 限制在区间：

$$
[1-\epsilon,\;1+\epsilon]
$$
里面。

如果：

$$
\epsilon=0.2
$$

那么 clip 区间就是：

$$
[0.8,\;1.2]
$$
| 原始 $\rho_t$ | clip 后 |
|---:|---:|
| 0.5 | 0.8 |
| 0.9 | 0.9 |
| 1.1 | 1.1 |
| 1.8 | 1.2 |

---

## 3.2 PPO-Clip 目标函数

PPO-Clip 的核心目标函数是：

$$
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\hat A_t,\,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
$$

这里有两个项：

第一项：

$$
\rho_t(\theta)\hat A_t
$$
是普通 policy gradient 的重要性采样修正版。

第二项：

$$
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
$$

是截断后的保守更新项。

PPO 取二者较小值：

$$
\min(\cdot,\cdot)
$$
目的不是让模型最快追逐 reward，而是防止策略一步更新太大。

---

## 3.3 clip 的直觉

当：

$$
\hat A_t>0
$$

说明这个动作好，模型想提高它的概率。  
但是 PPO 不允许概率无限提高。超过：

$$
1+\epsilon
$$
以后，继续提高概率不会带来额外收益。

当：

$$
\hat A_t<0
$$

说明这个动作不好，模型想降低它的概率。  
但是 PPO 也不允许概率无限降低。低于：

$$
1-\epsilon
$$
以后，继续降低概率也不会带来额外收益。

所以 clip 的作用是：

```text
给新旧策略的概率变化加安全边界，防止模型一次更新过猛。
```

---

# 4. PPO 的 value function 与 return target

PPO 通常是 actor-critic 结构。

- Actor：策略模型 $\pi_\theta(a \mid s)$
- Critic：价值模型 $V_\phi(s)$

Critic 的训练目标是让：

$$
V_\phi(s_t)
$$

接近：

$$
\hat R_t
$$
所以 value loss 可以写成：

$$
L_V(\phi)
=
\left(
V_\phi(s_t)-\hat R_t
\right)^2
$$

严格写优化问题时，应写：

$$
\phi^\star
=
\arg\min_\phi
\mathbb{E}_t
\left[
\left(
V_\phi(s_t)-\hat R_t
\right)^2
\right]
$$
这里要区分：

$$
\arg\min_\phi
$$

返回的是让 loss 最小的参数 $\phi^\star$。

而：

$$
\min_\phi
$$
返回的是最小 loss 值。

所以：

- 如果表达“求最优参数”，写 $\arg\min_\phi$；
- 如果表达“最小化这个 loss”，可以简写成 $\min_\phi L_V(\phi)$。

---

# 5. PPO 在大语言模型中的含义

在 LLM 里，可以对应成：

| 强化学习符号 | LLM 中的含义 |
|---|---|
| $s_t$ | prompt + 当前已经生成的 token 前缀 |
| $a_t$ | 下一个 token |
| $\pi_\theta(a_t\mid s_t)$ | 当前模型在这个上下文下生成某 token 的概率 |
| $r_t$ | token-level 或最终回答 reward |
| $V_\phi(s_t)$ | 当前前缀继续生成下去，最终能获得多少 reward 的估计 |
| $\hat R_t$ | 当前前缀对应的累计回报目标 |
| $\hat A_t$ | 这个 token 决策相对预期好多少 |

例如：

$$
s_t = (x,y_{<t})
$$

其中：

- $x$：用户 prompt；
- $y_{<t}$：已经生成的前面 token；
- $a_t=y_t$：当前要生成的 token。

PPO 在 LLM RLHF 中的难点是：  
为了估计每个 token 前缀的未来回报，需要训练 value model。这个 value model 显存开销高、训练不稳定，而且在长 CoT 场景下 credit assignment 仍然困难。

这正是 GRPO 想简化的地方。

---

# 6. GRPO 的核心思想：组内相对优势

GRPO 的核心是：

```text
不用 value model 判断回答好不好，而是对同一个 prompt 随机生成多个回答，把它们放在同一组里比较。
```

给定一个问题：

$$
q
$$
从旧策略模型中随机采样 $G$ 个回答：

$$
o_1,o_2,\dots,o_G
\sim
\pi_{\theta_{\text{old}}}(O\mid q)
$$

然后每个回答得到一个 reward：

$$
R_1,R_2,\dots,R_G
$$
或者在论文中常写成：

$$
r_1,r_2,\dots,r_G
$$

接着计算第 $i$ 个回答的组内相对 advantage：

$$
A_i=
\frac{
R_i-\operatorname{mean}(R_1,\dots,R_G)
}{
\operatorname{std}(R_1,\dots,R_G)
}
$$
含义是：

```text
第 i 个回答比同一题下其他回答平均水平好多少。
```

如果：

$$
A_i>0
$$

说明第 $i$ 个回答高于组内平均水平，应该提高它的生成概率。

如果：

$$
A_i<0
$$
说明第 $i$ 个回答低于组内平均水平，应该降低它的生成概率。

---

# 7. GRPO 中的随机采样、$R_i$ 与 $A_i$

## 7.1 GRPO 的样本不是固定答案，而是随机生成

前面例子中列出多个回答，容易让人误解为它们是预先准备好的样本。严格说，GRPO 的多个回答是从旧策略模型中随机采样出来的：

$$
o_i \sim \pi_{\theta_{\text{old}}}(o\mid q)
$$

完整回答是 token by token 生成的。

如果第 $i$ 个回答是：

$$
o_i=(y_{i,1},y_{i,2},\dots,y_{i,T})
$$
那么生成过程是：

$$
y_{i,1}\sim \pi_{\theta_{\text{old}}}(\cdot\mid q)
$$

$$
y_{i,2}\sim \pi_{\theta_{\text{old}}}(\cdot\mid q,y_{i,1})
$$
$$
y_{i,3}\sim \pi_{\theta_{\text{old}}}(\cdot\mid q,y_{i,1},y_{i,2})
$$

一直生成到结束。

完整回答概率是：

$$
\pi_{\theta_{\text{old}}}(o_i\mid q)
=
\prod_{t=1}^{T}
\pi_{\theta_{\text{old}}}(y_{i,t}\mid q,y_{i,<t})
$$
随机性来自模型采样过程，例如 temperature、top-p、top-k 等采样策略。

---

## 7.2 $R_i$ 是什么？

$$
R_i
$$

是第 $i$ 个回答的原始 reward 分数。

它不是概率。

它可以来自：

- 答案是否正确；
- 格式是否合格；
- 代码是否通过测试；
- 数学题最终答案是否匹配；
- reward model 打分；
- 教师规则或自动 verifier；
- 结构化输出 schema 是否符合要求。

例如同一道数学题生成 4 个回答：

| 回答 | 情况 | Reward |
|---|---|---:|
| $o_1$ | 答错 | $R_1=0$ |
| $o_2$ | 答对但格式不规范 | $R_2=0.7$ |
| $o_3$ | 答对且格式正确 | $R_3=1.0$ |
| $o_4$ | 答错 | $R_4=0$ |

---

## 7.3 $A_i$ 是什么？

$$
A_i
$$
是第 $i$ 个回答的 advantage。

在 GRPO 中，它来自组内 reward 标准化：

$$
A_i=
\frac{
R_i-\operatorname{mean}(R_1,\dots,R_G)
}{
\operatorname{std}(R_1,\dots,R_G)
}
$$

用上面的例子：

$$
R_1=0,\quad R_2=0.7,\quad R_3=1.0,\quad R_4=0
$$
平均值：

$$
\mu=\frac{0+0.7+1.0+0}{4}=0.425
$$

标准差近似：

$$
\sigma\approx 0.443
$$
那么：

$$
A_1=\frac{0-0.425}{0.443}\approx -0.96
$$

$$
A_2=\frac{0.7-0.425}{0.443}\approx 0.62
$$
$$
A_3=\frac{1.0-0.425}{0.443}\approx 1.30
$$

$$
A_4=\frac{0-0.425}{0.443}\approx -0.96
$$
解释：

- $o_3$：最好，强烈鼓励；
- $o_2$：高于平均，适度鼓励；
- $o_1,o_4$：低于平均，降低概率。

---

## 7.4 GRPO 的目标函数

GRPO 沿用 PPO 的概率比：

$$
\rho_i(\theta)
=
\frac{
\pi_\theta(o_i\mid q)
}{
\pi_{\theta_{\text{old}}}(o_i\mid q)
}
$$

然后使用 PPO-style clipped objective：

$$
\min
\left(
\rho_i(\theta)A_i,\,
\operatorname{clip}(\rho_i(\theta),1-\epsilon,1+\epsilon)A_i
\right)
$$
同时加入 reference model 的 KL 惩罚：

$$
-\beta D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\mathrm{ref}})
$$

简化后的目标可以写成：

$$
J_{\mathrm{GRPO}}(\theta)
=
\mathbb{E}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\left(
\min
\left(
\rho_i(\theta)A_i,\,
\operatorname{clip}(\rho_i(\theta),1-\epsilon,1+\epsilon)A_i
\right)
-
\beta D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\mathrm{ref}})
\right)
\right]
$$
这里：

- $\rho_i(\theta)$：新旧策略对第 $i$ 个回答的概率比；
- $A_i$：组内相对 advantage；
- $\epsilon$：clip 范围；
- $\beta$：KL 惩罚系数；
- $\pi_{\mathrm{ref}}$：参考模型，通常是 SFT model 或 base model。

---

# 8. PPO 与 GRPO 的完整对比

| 维度 | PPO | GRPO |
|---|---|---|
| 算法类型 | on-policy policy gradient | PPO 变体，critic-free policy gradient |
| 是否需要 critic / value model | 通常需要 | 不需要 |
| advantage 来源 | $\hat R_t - V_\phi(s_t)$ 或 GAE | 同一 prompt 下多个回答的组内相对 reward |
| 核心更新 | clipped surrogate objective | 仍然是 clipped surrogate objective |
| 是否随机采样 | 是，从策略与环境交互采样轨迹 | 是，同一 prompt 从旧策略随机采样多个回答 |
| 训练成本 | actor + critic，显存较重 | 去掉 critic，但需要多回答 rollout |
| 适合场景 | 通用 RL、机器人、游戏、RLHF | 数学、代码、逻辑推理、答案可验证任务 |
| 主要风险 | critic 不准影响 advantage | reward 方差小、全对/全错时学习信号弱 |
| credit assignment | critic 可尝试 token/state-level 估计 | 主要是整条输出级别的相对好坏 |
| 工程难点 | value model、GAE、KL、稳定性 | group size、采样温度、reward 设计、KL、长度控制 |

一句话对比：

```text
PPO 用 critic 判断“这个动作是否比预期更好”；
GRPO 用同题多个回答判断“这个回答是否比同组其他回答更好”。
```

---

# 9. GRPO 的主要不足

GRPO 的优势是省掉 critic，降低显存和实现复杂度，适合答案可验证的任务。但它也有明显不足。

## 9.1 对 reward / verifier 依赖强

GRPO 的优化方向完全由 reward 决定。

如果 reward 只检查最终答案，模型可能学会猜答案。  
如果 reward 只检查格式，模型可能学会格式正确但内容空洞。  
如果 reward model 有漏洞，模型可能 reward hacking。

---

## 9.2 组内 reward 方差太小时，学习信号弱

GRPO 的 advantage 依赖：

$$
\operatorname{std}(R_1,\dots,R_G)
$$

如果同一组回答全部答错：

$$
R_1=R_2=\cdots=R_G=0
$$
那么组内没有区分度。

如果全部答对：

$$
R_1=R_2=\cdots=R_G=1
$$

同样也没有区分度。

所以 GRPO 最喜欢的样本是：

```text
错、半对、对、格式好
```

这种组内有差异的样本。

这意味着：数据难度分布很重要。太难或太简单的题都不利于 GRPO 学习。

---

## 9.3 样本效率不一定高

GRPO 省掉了 critic，但每个 prompt 要采样多个回答。

如果：

$$
G=8
$$
那么一个问题要生成 8 条完整回答。

如果每条回答都是长 CoT，rollout 成本会很高。

所以 GRPO 的成本不是消失了，而是从：

```text
训练 critic 的成本
```

转移到了：

```text
多样本生成 + 长 CoT rollout 的成本
```

---

## 9.4 长链推理 credit assignment 弱

如果 reward 只给整条回答一个分数，例如：

$$
R_i=1
$$

模型知道这条回答好，但不知道具体是哪一步推理导致好。

如果：

$$
R_i=0
$$
模型知道这条回答差，但不知道哪一步开始错。

所以 GRPO 擅长判断：

```text
哪条回答更好
```

但不擅长判断：

```text
这条回答里面哪一步贡献最大
```

---

## 9.5 对超参数敏感

GRPO 对这些参数都敏感：

- group size；
- temperature；
- top-p；
- KL 系数；
- clip 范围；
- reward 权重；
- 长度惩罚；
- batch 构造方式。

如果 temperature 太低，采样太相似，组内方差小。  
如果 temperature 太高，采样质量差，reward 噪声大。  
如果 KL 太弱，模型容易漂移。  
如果 KL 太强，模型学不动。

---

## 9.6 容易出现风格退化

如果 reward 主要奖励答案正确，模型可能牺牲：

- 可读性；
- 语言一致性；
- 结构清晰度；
- 用户体验；
- 回答简洁性。

DeepSeek-R1-Zero 就出现过 poor readability 和 language mixing，因此后续 DeepSeek-R1 加入 cold-start data、SFT、rejection sampling 和多阶段 RL 来修正。

---

# 10. 当前比原始 GRPO 更值得关注的替代或改进方法

这里不是说某个方法绝对优于 GRPO，而是不同场景有更合适的选择。

| 方法 | 适合场景 | 与 GRPO 的关系 |
|---|---|---|
| DAPO | 数学、代码、长 CoT、可验证推理 | 在 GRPO 基础上改进采样、clip、token-level loss、过长惩罚 |
| Dr.GRPO | 已有 GRPO pipeline，希望低成本修正长度/聚合偏置 | GRPO 的偏置修正版 |
| GSPO | MoE 模型、长序列、sequence-level reward | 从 token-level importance ratio 转向 sequence-level optimization |
| VAPO | 算力充足、追求更强长链推理 | 重新引入 value signal，改善 credit assignment |
| RLOO / REINFORCE++ | 想要 critic-free，但不一定用 group relative | PPO 的轻量替代路线 |
| DPO / Online DPO | 有偏好数据，没有可靠 verifier | 不是在线 RLVR，更适合偏好对齐 |

简单选择：

```text
答案可验证：优先看 GRPO / DAPO / Dr.GRPO / GSPO / VAPO
只有偏好数据：优先看 DPO / Online DPO
算力有限：优先 SFT + DPO + 小规模 GRPO / Dr.GRPO
追求复杂推理上限：考虑 DAPO / VAPO / GSPO
```

---

# 11. DeepSeek 系列与 GRPO 的关系

## 11.1 DeepSeekMath

DeepSeekMath 论文中提出 GRPO，并明确称它是 PPO 的一个变体，用于增强数学推理能力，同时优化 PPO 的显存使用。

## 11.2 DeepSeek-R1 / R1-Zero

DeepSeek-R1 论文说明：

- DeepSeek-R1-Zero 使用 GRPO 作为 RL 算法；
- 使用 rule-based reward，包括 accuracy reward 和 format reward；
- DeepSeek-R1 在后续阶段继续使用多阶段流程，包括 cold-start data、SFT、rejection sampling 和 RL。

## 11.3 DeepSeek-V4-Pro

按 DeepSeek-V4-Pro 官方 Hugging Face 模型卡，DeepSeek-V4 系列是先做大规模预训练，然后进行综合 post-training。其 post-training 包含：

```text
domain-specific experts through SFT and RL with GRPO,
followed by unified model consolidation via on-policy distillation
```

因此，严谨说法是：

```text
DeepSeek-V4-Pro 不是“整个训练都靠 GRPO”。
它的预训练不是 GRPO；
但它的 post-training / RL 阶段使用了 GRPO。
```

---

# 12. 一页速记版

## PPO

```text
PPO = 策略梯度 + 新旧策略概率比 + clip 限制 + critic/value baseline
```

关键公式：

$$
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\theta_{\text{old}}}(a_t\mid s_t)
}
$$

$$
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\hat A_t,\,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
$$
$$
\hat A_t=\hat R_t - V_\phi(s_t)
$$

$$
\phi^\star
=
\arg\min_\phi
\mathbb{E}_t
\left[
\left(
V_\phi(s_t)-\hat R_t
\right)^2
\right]
$$
---

## GRPO

```text
GRPO = PPO-Clip + group-relative advantage - critic/value model
```

采样：

$$
o_1,\dots,o_G\sim\pi_{\theta_{\text{old}}}(O\mid q)
$$

reward：

$$
R_1,\dots,R_G
$$
advantage：

$$
A_i=
\frac{
R_i-\operatorname{mean}(R_1,\dots,R_G)
}{
\operatorname{std}(R_1,\dots,R_G)
}
$$

更新：

```text
A_i > 0：提高该回答概率
A_i < 0：降低该回答概率
```

---

## 最核心区别

```text
PPO：用 value model 判断“比预期好不好”。
GRPO：用同题多回答比较“比同组其他回答好不好”。
```

---

# 13. 参考资料

1. Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.  
   https://arxiv.org/abs/1707.06347

2. Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.  
   https://arxiv.org/abs/2402.03300

3. Guo et al. (2025). *DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning*. Nature.  
   https://www.nature.com/articles/s41586-025-09422-z

4. DeepSeek-V4-Pro Official Hugging Face Model Card.  
   https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro

5. Hugging Face TRL GRPO Trainer Documentation.  
   https://huggingface.co/docs/trl/grpo_trainer

---

# 附：面向 K12 题目解析系统的训练建议

如果目标是 K12 题目解析、错题诊断、知识点标注，不建议直接只用原始 GRPO。更合理路线是：

```text
SFT → DPO / 偏好优化 → DAPO 或 Dr.GRPO → 需要时再考虑 VAPO
```

reward 不应只看最终答案，而应拆成多维：

$$
R =
R_{\text{答案正确}}
+
R_{\text{步骤合理}}
+
R_{\text{知识点匹配}}
+
R_{\text{格式规范}}
+
R_{\text{难度一致}}
$$

否则模型容易只优化最终答案，而不真正提升解析质量。
