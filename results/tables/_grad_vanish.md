# 物理约束"零效果"的机制定位（只读诊断，不改任何训练产物）

## 结论摘要（含一次自我更正）

最初假设是"`λ·L_freq` 值很大但梯度被 softmax 饱和吃掉（浮点意义上精确为 0）"。
**这个假设被数据推翻了**：实测 softmax 在 float32 下精确 one-hot 的样本占比是 **0.0%**，
梯度并非精确零。

真正的首要机制是 **λ 调度与"最优 epoch"错配**：
`stepwise` 策略下 `λ(t)=0` 当 `t < warmup_epochs=20`，
而验证指标在 **epoch 19 附近就已见顶**（1-shot 下模型 9 轮就 `train_acc=1.000`，随后在 val 上过拟合下行）。
于是被选中的 `best.pt` 取自 **λ 仍为 0 的那一轮** —— 物理项对它没有任何影响。

| 证据 | 问了什么问题 | 结论 |
|---|---|---|
| **C（首要）** | λ 生效的那一轮，是否早于最优 epoch？ | 多数种子 **否**，best_epoch=19 < warmup=20 |
| A | 两组 `best.pt` 权重差多少？ | best_epoch<20 的种子**逐位相同**；其余有微小差异 |
| B | 物理项的参数梯度有多大？ | 相对 CE 小 2~3 个数量级（次要因素） |

## 证据 C（首要）：λ 调度 vs best_epoch 错配

逐 (方法, 种子) 从 `runs_<tag>.csv` 读 `best_epoch`，再用 `StepwiseLambda(warmup=20, step=10, …)` 算出该轮的 λ。
若 `λ(best_epoch) == 0`，说明被选中的权重里**不含物理项的任何贡献**。

| tag | method | seed | best_epoch | λ(best_epoch) | best_val_f1 | 物理项是否影响该权重 |
|---|---|---|---|---|---|---|
| bnd3c_snrm8_lam5 | CNN-CE | 0 | 19 | 0.00 | 0.9381 | —（CE 基线） |
| bnd3c_snrm8_lam5 | CNN-CE | 1 | 19 | 0.00 | 0.8979 | —（CE 基线） |
| bnd3c_snrm8_lam5 | CNN-CE | 2 | 59 | 0.40 | 0.9725 | —（CE 基线） |
| bnd3c_snrm8_lam5 | CNN-Freq | 0 | 19 | 0.00 | 0.9381 | **否（λ=0，权重与 CE 同源）** |
| bnd3c_snrm8_lam5 | CNN-Freq | 1 | 19 | 0.00 | 0.8979 | **否（λ=0，权重与 CE 同源）** |
| bnd3c_snrm8_lam5 | CNN-Freq | 2 | 59 | 0.40 | 0.9725 | 是 |
| bnd4c_snrm8_lam5 | CNN-CE | 0 | 39 | 0.20 | 0.7986 | —（CE 基线） |
| bnd4c_snrm8_lam5 | CNN-CE | 1 | 39 | 0.20 | 0.8014 | —（CE 基线） |
| bnd4c_snrm8_lam5 | CNN-CE | 2 | 59 | 0.40 | 0.9688 | —（CE 基线） |
| bnd4c_snrm8_lam5 | CNN-Freq | 0 | 79 | 0.50 | 0.7922 | 是 |
| bnd4c_snrm8_lam5 | CNN-Freq | 1 | 39 | 0.20 | 0.7742 | 是 |
| bnd4c_snrm8_lam5 | CNN-Freq | 2 | 59 | 0.40 | 0.9688 | 是 |

⇒ `CNN-Freq` 的 6 组里，有 **2 组**的 `best_epoch` 落在 λ=0 区间。

**这直接给出了实验设计的修法**：把 λ 提到目标值的时间点挪到**验证峰之前**，
例如 `warmup_epochs=5, step=2`（λ 在 ~epoch 15 到 0.5），
或改用 `fixed`（λ 恒为 0.5）配合短 warmup。
否则无论 L_freq 设计得多好，它都不会出现在被报告的模型里。

## 证据 A：best.pt 权重差异（CE vs Freq，同种子）

⚠️ `max\|Δθ\|` 会被 **BatchNorm 的 `num_batches_tracked` 计数器**污染：它是"该 BN 层前向次数"，取决于 `best_epoch`。
若两方法的最优 epoch 不同，这个计数器能差出上千，表现为 `max\|Δθ\|=2e3` 这种假警报（见下方张量明细）。

| tag | seed | max\|Δθ\| | 相对 L2 | 变动参数数/总数 | 判定 |
|---|---|---|---|---|---|
| bnd3c_snrm8_lam5 | 0 | 0.000e+00 | 0.000e+00 | 0/46087 | **完全一致（物理项零贡献）** |
| bnd3c_snrm8_lam5 | 1 | 0.000e+00 | 0.000e+00 | 0/46087 | **完全一致（物理项零贡献）** |
| bnd3c_snrm8_lam5 | 2 | 1.297e-02 | 8.538e-06 | 44856/46087 | 确有差异 |
| bnd4c_snrm8_lam5 | 0 | 2.000e+03 | 1.000e+00 | 46152/46152 | 确有差异 |
| bnd4c_snrm8_lam5 | 1 | 1.036e-01 | 1.235e-04 | 44357/46152 | 确有差异 |
| bnd4c_snrm8_lam5 | 2 | 1.423e-02 | 6.186e-06 | 44945/46152 | 确有差异 |

差异最大的 5 个张量（用于排查是不是单个异常 buffer 撑起了 `max\|Δθ\|`）：

| tag | seed | 张量 | max\|Δ\| | 该张量 max\|θ\| |
|---|---|---|---|---|
| bnd3c_snrm8_lam5 | 0 | `blocks.0.conv.weight` | 0.000e+00 | 2.669e-01 |
| bnd3c_snrm8_lam5 | 0 | `blocks.0.conv.bias` | 0.000e+00 | 2.494e-01 |
| bnd3c_snrm8_lam5 | 0 | `blocks.0.bn.weight` | 0.000e+00 | 1.015e+00 |
| bnd3c_snrm8_lam5 | 0 | `blocks.0.bn.bias` | 0.000e+00 | 1.714e-02 |
| bnd3c_snrm8_lam5 | 0 | `blocks.0.bn.running_mean` | 0.000e+00 | 2.531e-01 |
| bnd3c_snrm8_lam5 | 1 | `blocks.0.conv.weight` | 0.000e+00 | 2.615e-01 |
| bnd3c_snrm8_lam5 | 1 | `blocks.0.conv.bias` | 0.000e+00 | 2.411e-01 |
| bnd3c_snrm8_lam5 | 1 | `blocks.0.bn.weight` | 0.000e+00 | 1.019e+00 |
| bnd3c_snrm8_lam5 | 1 | `blocks.0.bn.bias` | 0.000e+00 | 1.304e-02 |
| bnd3c_snrm8_lam5 | 1 | `blocks.0.bn.running_mean` | 0.000e+00 | 2.418e-01 |
| bnd3c_snrm8_lam5 | 2 | `blocks.3.bn.running_mean` | 1.297e-02 | 1.092e+00 |
| bnd3c_snrm8_lam5 | 2 | `fc2.weight` | 1.104e-02 | 2.053e-01 |
| bnd3c_snrm8_lam5 | 2 | `blocks.3.bn.running_var` | 4.490e-03 | 4.775e-01 |
| bnd3c_snrm8_lam5 | 2 | `blocks.1.bn.running_mean` | 4.463e-03 | 1.296e+00 |
| bnd3c_snrm8_lam5 | 2 | `blocks.2.bn.running_mean` | 4.056e-03 | 7.498e-01 |
| bnd4c_snrm8_lam5 | 0 | `blocks.0.bn.num_batches_tracked` | 2.000e+03 | 2.000e+03 |
| bnd4c_snrm8_lam5 | 0 | `blocks.1.bn.num_batches_tracked` | 2.000e+03 | 2.000e+03 |
| bnd4c_snrm8_lam5 | 0 | `blocks.2.bn.num_batches_tracked` | 2.000e+03 | 2.000e+03 |
| bnd4c_snrm8_lam5 | 0 | `blocks.3.bn.num_batches_tracked` | 2.000e+03 | 2.000e+03 |
| bnd4c_snrm8_lam5 | 0 | `blocks.2.bn.running_mean` | 1.469e-01 | 1.116e+00 |
| bnd4c_snrm8_lam5 | 1 | `blocks.3.bn.running_mean` | 1.036e-01 | 1.206e+00 |
| bnd4c_snrm8_lam5 | 1 | `blocks.2.bn.running_mean` | 4.889e-02 | 1.021e+00 |
| bnd4c_snrm8_lam5 | 1 | `fc2.weight` | 3.958e-02 | 2.376e-01 |
| bnd4c_snrm8_lam5 | 1 | `blocks.1.bn.running_mean` | 3.213e-02 | 8.413e-01 |
| bnd4c_snrm8_lam5 | 1 | `fc1.weight` | 2.339e-02 | 2.276e-01 |
| bnd4c_snrm8_lam5 | 2 | `fc2.weight` | 1.423e-02 | 2.962e-01 |
| bnd4c_snrm8_lam5 | 2 | `blocks.3.bn.running_mean` | 7.342e-03 | 1.313e+00 |
| bnd4c_snrm8_lam5 | 2 | `blocks.1.bn.running_mean` | 3.213e-03 | 1.166e+00 |
| bnd4c_snrm8_lam5 | 2 | `blocks.3.bn.running_var` | 2.574e-03 | 4.650e-01 |
| bnd4c_snrm8_lam5 | 2 | `blocks.2.bn.running_mean` | 2.539e-03 | 8.420e-01 |

## 证据 B（次要）：L_freq 的参数梯度量级

证据 C 说明"大多数种子根本没被物理项碰过"；本节的 B-1 则回答另一半问题：
**万一 λ 生效了（如 best_epoch=59 的种子），它到底有多大力气？**

### bnd3c_snrm8_lam5（3 类；权重取自 bnd3c_snrm8_lam5 的 CNN-Freq_seed0）

**B-1 参数梯度对比**（真实工作点，λ=0.5，前 600 个 test 窗口）：

| 量 | 数值 |
|---|---|
| L_CE | 2.562e+00 |
| L_freq（未乘 λ） | 0.0975 |
| ‖∂L_CE/∂θ‖ | 2.543e+01 |
| ‖λ·∂L_freq/∂θ‖（λ=0.5） | 4.035e-02 |
| **比值 ‖λ∇L_freq‖ / ‖∇L_CE‖** | **1.587e-03** |
| logit margin 中位数 / P10 | 9.74 / 5.93 |
| softmax 在 float32 下**恰好** one-hot 的样本占比 | **0.0%** |

→ softmax 只有 0.0% 精确 one-hot ⇒ **"梯度被浮点精确截断为 0"这个假设不成立**。梯度真实但很小，所以真正原因得往调度/选点上找（见证据 C）。

⚠️ 口径说明：`L_CE` 是在 **test 池窗口**上算的（异分布，故达 2.56），不等于训练时已饱和的 query CE（≈1e-4），因此该比值不等于训练时的比值，只作为"物理项梯度比 CE 小 2~3 个数量级"这一**量级**结论的依据。

**B-2 梯度量级 vs logit 饱和程度**：把 logits 乘缩放因子 `s`，`s=1` 是真实工作点，`s<1` 人为解除饱和。
| logits 缩放 s | L_freq | max\|∂L/∂logits\| | mean\|∂L/∂logits\| | logit margin |
|---|---|---|---|---|
| 1.00 | 0.0975 | 8.452e-05 | 6.817e-06 | 9.74 |
| 0.50 | 0.1014 | 3.145e-04 | 3.407e-05 | 4.87 |
| 0.25 | 0.1062 | 5.055e-04 | 6.458e-05 | 2.44 |
| 0.10 | 0.1112 | 5.856e-04 | 8.774e-05 | 0.97 |
| 0.05 | 0.1135 | 5.983e-04 | 9.362e-05 | 0.49 |

> s 从 1.0 降到 0.05（logit margin 从 9.74 降到 0.49），L_freq 值只从 0.0975 变到 0.1135 —— **几乎不变**。
> 而梯度范数放大约 **7 倍**，方向一致：**饱和是主因**——L_freq 在饱和区几乎是个常数，代价看得见、梯度看不见。

### bnd4c_snrm8_lam5（4 类；权重取自 bnd4c_snrm8_lam5 的 CNN-Freq_seed0）

**B-1 参数梯度对比**（真实工作点，λ=0.5，前 600 个 test 窗口）：

| 量 | 数值 |
|---|---|
| L_CE | 1.046e+00 |
| L_freq（未乘 λ） | 0.1550 |
| ‖∂L_CE/∂θ‖ | 2.457e+01 |
| ‖λ·∂L_freq/∂θ‖（λ=0.5） | 1.003e+00 |
| **比值 ‖λ∇L_freq‖ / ‖∇L_CE‖** | **4.082e-02** |
| logit margin 中位数 / P10 | 2.02 / 0.59 |
| softmax 在 float32 下**恰好** one-hot 的样本占比 | **0.0%** |

→ softmax 只有 0.0% 精确 one-hot ⇒ **"梯度被浮点精确截断为 0"这个假设不成立**。梯度真实但很小，所以真正原因得往调度/选点上找（见证据 C）。

⚠️ 口径说明：`L_CE` 是在 **test 池窗口**上算的（异分布，故达 1.05），不等于训练时已饱和的 query CE（≈1e-4），因此该比值不等于训练时的比值，只作为"物理项梯度比 CE 小 2~3 个数量级"这一**量级**结论的依据。

**B-2 梯度量级 vs logit 饱和程度**：把 logits 乘缩放因子 `s`，`s=1` 是真实工作点，`s<1` 人为解除饱和。
| logits 缩放 s | L_freq | max\|∂L/∂logits\| | mean\|∂L/∂logits\| | logit margin |
|---|---|---|---|---|
| 1.00 | 0.1550 | 3.421e-04 | 5.630e-05 | 2.02 |
| 0.50 | 0.1564 | 4.326e-04 | 8.943e-05 | 1.01 |
| 0.25 | 0.1532 | 5.188e-04 | 1.089e-04 | 0.51 |
| 0.10 | 0.1501 | 5.212e-04 | 1.167e-04 | 0.20 |
| 0.05 | 0.1492 | 5.009e-04 | 1.180e-04 | 0.10 |

> s 从 1.0 降到 0.05（logit margin 从 2.02 降到 0.10），L_freq 值只从 0.1550 变到 0.1492 —— **几乎不变**。
> 梯度范数只放大约 **1.5 倍**（不是量级变化），说明饱和只是次要因素：更主要的原因是**L_freq 对 logits 本身就不敏感**（rel 数值区间窄、类间差异小）。两条原因叠加，结果都是"损失值可见、优化信号不可见"。

