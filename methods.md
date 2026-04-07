

# 方法对照表（完整 loss 版）

## 1) 统一定义

设一个 batch 中有 $B$ 个 sampled responses，第 $i$ 个样本对应：

- prompt: $x_i$
- response: $y_i=(y_{i,1},\dots,y_{i,T_i})$
- response token 位置集合（mask）: $M_i\subseteq \{1,\dots,T_i\}$
- response 长度: $|M_i|$

记整个 batch 的 response token 总数为：
$$
|M|=\sum_{i=1}^{B}|M_i|.
$$

---

### 1.1 旧策略 / 当前策略

- rollout 时旧策略：$\pi_{\text{old}}$
- 当前更新策略：$\pi_\theta$

token 级重要性比率定义为：
$$
r_{i,t}(\theta)
=
\frac{\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}{\pi_{\text{old}}(y_{i,t}\mid x_i,y_{i,<t})}
=
\exp\!\left(
\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})
-
\log\pi_{\text{old}}(y_{i,t}\mid x_i,y_{i,<t})
\right).
$$

---

### 1.2 基础序列奖励

基础 reward 为二值序列奖励：
$$
r_i = r(x_i,y_i)\in\{0,1\}.
$$

在当前实现中：
- 格式正确且答案正确：$r_i=1$
- 其他情况：$r_i=0$

---

### 1.3 组内优势（GRPO baseline）

对同一个 prompt 的 $G$ 条采样，记组内奖励为 $\{r_j\}_{j=1}^{G}$，组均值为：
$$
\bar r = \frac{1}{G}\sum_{j=1}^{G}r_j.
$$

则序列级组内优势为：
$$
A_i^{\text{seq}}=r_i-\bar r.
$$

如果开启 `use_std_normalization`，则：
$$
A_i^{\text{seq}}
=
\frac{r_i-\bar r}{\operatorname{std}(r_{1:G})+\epsilon}.
$$

若不开 step reward，通常将它广播到该 response 的所有 token：
$$
A_{i,t}=A_i^{\text{seq}},\qquad t\in M_i.
$$

---

### 1.4 overlong penalty（DAPO 组件）

对第 $i$ 个 response，若长度为 $L_i$，目标长度为 $L_0$，则长度惩罚定义为：
$$
\operatorname{penalty}(L_i)
=
\lambda_{\text{len}} \cdot \frac{\max(L_i-L_0,0)}{L_0}.
$$

惩罚后的 reward 为：
$$
r_i' = r_i-\operatorname{penalty}(L_i).
$$

随后用 $\{r_j'\}$ 重新计算组内优势：
$$
\bar r'=\frac{1}{G}\sum_{j=1}^{G}r_j',
\qquad
A_i^{\text{seq, overlong}}=r_i'-\bar r'.
$$

若开 std-normalization，则进一步：
$$
A_i^{\text{seq, overlong}}
=
\frac{r_i'-\bar r'}{\operatorname{std}(r_1',\dots,r_G')+\epsilon}.
$$

---

### 1.5 step progress reward

将 reasoning 切分成步骤前缀 $s_{i,1},\dots,s_{i,K_i}$，估计每一步后的通过率：
$$
p_{i,k}\approx \Pr(\text{correct}\mid x_i,s_{i,k}),
\qquad
p_{i,0}\approx \Pr(\text{correct}\mid x_i).
$$

定义第 $k$ 步的即时进展奖励：
$$
\delta_{i,k}=p_{i,k}-p_{i,k-1}.
$$

定义 discounted $\lambda$-return：
$$
R_{i,k}^{\lambda} = \delta_{i,k}+\gamma\lambda R_{i,k+1}^{\lambda}.
$$

将 $R_{i,k}^{\lambda}$ 映射回步骤覆盖的 response tokens，得到 token 级步骤优势 $A_{i,t}^{\text{step}}$。

最终 token 级优势为：
$$
A_{i,t}^{\text{final}}
=
A_i^{\text{seq}}+\alpha A_{i,t}^{\text{step}}.
$$

若同时使用 overlong penalty，则把 $A_i^{\text{seq}}$ 换成 $A_i^{\text{seq, overlong}}$。

---

### 1.6 KL 近似项

当前实现支持三种 token 级近似 KL：
$$
\widehat{\mathrm{KL}}_{i,t}^{(k1)}=\log r_{i,t},
$$
$$
\widehat{\mathrm{KL}}_{i,t}^{(k2)}=\frac12(\log r_{i,t})^2,
$$
$$
\widehat{\mathrm{KL}}_{i,t}^{(k3)}=r_{i,t}-1-\log r_{i,t}.
$$

默认使用 `k3`。统一记为：
$$
\widehat{\mathrm{KL}}_{i,t}.
$$

---

### 1.7 两种 batch 聚合方式

#### token aggregation
$$
\operatorname{Agg}_{\text{tok}}(\ell)
=
\frac{1}{|M|} \sum_{i=1}^{B}\sum_{t\in M_i}\ell_{i,t}.
$$

#### seq aggregation
$$
\operatorname{Agg}_{\text{seq}}(\ell)
=
\frac{1}{B} \sum_{i=1}^{B} \frac{1}{|M_i|} \sum_{t\in M_i}\ell_{i,t}.
$$

后文统一记为：
$$
\operatorname{Agg}(\ell),
$$
其具体是 `token` 还是 `seq`，由配置 `loss_aggregation` 决定。

---

## 2) 方法对照表（表格版）

> 说明：表中先写 token 级损失 $\ell_{i,t}$，再写 batch loss $\mathcal L=\operatorname{Agg}(\ell)$。

| 方法 | token 级损失 $\ell_{i,t}$ | 完整 batch loss $\mathcal L$ | 说明 |
|------|---------------------------|-------------------------------|------|
| **no_baseline** | $\ell_{i,t}=- r_i \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})$ | $\mathcal L_{\text{no-baseline}}=\operatorname{Agg}\!\left(- r_i \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})\right)$ | 原始 REINFORCE，直接用原始序列 reward 做权重 |
| **reinforce_with_baseline** | $\ell_{i,t}=- A_{i,t} \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})$ | $\mathcal L_{\text{reinforce}}=\operatorname{Agg}\!\left(- A_{i,t} \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})\right)$ | 通常 $A_{i,t}=A_i^{\text{seq}}$ |
| **grpo_no_clip** | $\ell_{i,t}=- A_{i,t} r_{i,t}$ | $\mathcal L_{\text{grpo-noclip}}=\operatorname{Agg}\!\left(- A_{i,t} r_{i,t}\right)$ | importance-weighted surrogate |
| **grpo_clip** | $\ell_{i,t}=-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\right)$ | $\mathcal L_{\text{grpo-clip}}=\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\right)\right)$ | PPO/GRPO 标准 clip |
| **grpo_clip + KL** | $\ell_{i,t}=-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\right)+\beta \widehat{\mathrm{KL}}_{i,t}$ | $\mathcal L_{\text{grpo-clip-kl}}=\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\right)+\beta \widehat{\mathrm{KL}}_{i,t}\right)$ | 标准 clip 上叠加 KL 正则 |
| **dapo_clip** | $\ell_{i,t}=-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\operatorname{clip}(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\right)$ | $\mathcal L_{\text{dapo}}=\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\operatorname{clip}(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\right)\right)$ | 上下界解耦 clip |
| **dapo_clip + overlong** | $\ell_{i,t}=-\min\!\left(A_{i,t}^{\text{overlong}}r_{i,t},\; A_{i,t}^{\text{overlong}}\operatorname{clip}(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\right)$ | $\mathcal L_{\text{dapo-overlong}}=\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}^{\text{overlong}}r_{i,t},\; A_{i,t}^{\text{overlong}}\operatorname{clip}(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\right)\right)$ | $A_{i,t}^{\text{overlong}}=A_i^{\text{seq, overlong}}$ |
| **dapo_clip + filter** | 与 `dapo_clip` 相同 | $\mathcal L_{\text{dapo-filter}}=\operatorname{Agg}_{(x_i,y_i)\sim \mathcal D_{\text{kept}}}\!\left(-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\operatorname{clip}(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\right)\right)$ | 公式不变，仅训练分布被过滤 |
| **dapo full** | $\ell_{i,t}=-\min\!\left(A_{i,t}^{\text{overlong}}r_{i,t},\; A_{i,t}^{\text{overlong}}\operatorname{clip}(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\right)$ | $\mathcal L_{\text{dapo-full}}=\operatorname{Agg}_{(x_i,y_i)\sim \mathcal D_{\text{kept}}}\!\left(-\min\!\left(A_{i,t}^{\text{overlong}}r_{i,t},\; A_{i,t}^{\text{overlong}}\operatorname{clip}(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\right)\right)$ | decoupled clip + filter + penalty |
| **gbpo_clip** | $\ell_{i,t}=-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}\right)$ | $\mathcal L_{\text{gbpo}}=\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}\right)\right)$ | $\widetilde r_{i,t}$ 随符号不同 clip |
| **gbpo_clip + KL** | $\ell_{i,t}=-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}\right)+\beta \widehat{\mathrm{KL}}_{i,t}$ | $\mathcal L_{\text{gbpo-kl}}=\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}\right)+\beta \widehat{\mathrm{KL}}_{i,t}\right)$ | sign-aware clip + KL |
| **grpo_clip + step progress** | $\ell_{i,t}=-\min\!\left(A_{i,t}^{\text{final}}r_{i,t},\; A_{i,t}^{\text{final}}\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\right)$ | $\mathcal L_{\text{step+clip}}=\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}^{\text{final}}r_{i,t},\; A_{i,t}^{\text{final}}\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\right)\right)$ | $A_{i,t}^{\text{final}}=A_i^{\text{seq}}+\cdots$ |
| **grpo_clip + step + KL** | $\ell_{i,t}=-\min\!\left(A_{i,t}^{\text{final}}r_{i,t},\; A_{i,t}^{\text{final}}\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\right)+\beta \widehat{\mathrm{KL}}_{i,t}$ | $\mathcal L_{\text{step+clip+kl}}=\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}^{\text{final}}r_{i,t},\; A_{i,t}^{\text{final}}\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\right)+\beta \widehat{\mathrm{KL}}_{i,t}\right)$ | 组合 step 优势与 KL |
| **OPSD** | $\ell_{i,t}=k\cdot \operatorname{KL}\!\Big(p_\theta^t(\cdot\mid x_i^{\text{teacher}},y_{i,<t}) \parallel p_\theta^s(\cdot\mid x_i,y_{i,<t})\Big)$ | $\mathcal L_{\text{opsd}}=\frac{1}{|M|}\sum_{i=1}^{B}\sum_{t\in M_i}\ell_{i,t}$ | teacher/student 双前向 KL |
| **EI / SFT** | $\ell_{i,t}=-\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})$ | $\mathcal L_{\text{EI-SFT}}=\frac{1}{|M|}\sum_{i=1}^{B}\sum_{t\in M_i}-\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})$ | 仅在 expert rollout 上训练 |

---

## 3) 各方法完整展开式

### 3.1 no_baseline

token 级损失：
$$
\ell_{i,t}^{\text{no-baseline}}
=
- r_i \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

batch loss：
$$
\mathcal L_{\text{no-baseline}}
=
\operatorname{Agg}\!\left(- r_i \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})\right).
$$

token aggregation 显式：
$$
\mathcal L_{\text{no-baseline}}^{\text{tok}}
=
\frac{1}{|M|} \sum_{i=1}^{B}\sum_{t\in M_i}
- r_i \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

seq aggregation 显式：
$$
\mathcal L_{\text{no-baseline}}^{\text{seq}}
=
\frac{1}{B} \sum_{i=1}^{B} \frac{1}{|M_i|} \sum_{t\in M_i}
- r_i \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

---

### 3.2 reinforce_with_baseline

$$
A_{i,t}=A_i^{\text{seq}}
\quad\text{or}\quad
A_{i,t}=\frac{r_i-\bar r}{\operatorname{std}(r_{1:G})+\epsilon}.
$$

token 级损失：
$$
\ell_{i,t}^{\text{reinforce}}
=
- A_{i,t} \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

batch loss：
$$
\mathcal L_{\text{reinforce}}
=
\operatorname{Agg}\!\left(- A_{i,t} \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})\right).
$$

---

### 3.3 grpo_no_clip

$$
\ell_{i,t}^{\text{grpo-noclip}}
=
- A_{i,t} r_{i,t}.
$$
$$
\mathcal L_{\text{grpo-noclip}}
=
\operatorname{Agg}\!\left(- A_{i,t} r_{i,t}\right).
$$

显式：
$$
\mathcal L_{\text{grpo-noclip}}^{\text{tok}}
=
\frac{1}{|M|} \sum_{i=1}^{B}\sum_{t\in M_i} - A_{i,t} r_{i,t},
$$
$$
\mathcal L_{\text{grpo-noclip}}^{\text{seq}}
=
\frac{1}{B} \sum_{i=1}^{B} \frac{1}{|M_i|} \sum_{t\in M_i} - A_{i,t} r_{i,t}.
$$

---

### 3.4 grpo_clip

令
$$
\widetilde r_{i,t}^{\text{clip}}
=
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon).
$$

则
$$
\ell_{i,t}^{\text{grpo-clip}}
=
-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{clip}}\right),
$$
$$
\mathcal L_{\text{grpo-clip}}
=
\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{clip}}\right)\right).
$$

---

### 3.5 grpo_clip + KL

$$
\ell_{i,t}^{\text{grpo-clip-kl}}
=
-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{clip}}\right)
+\beta \widehat{\mathrm{KL}}_{i,t},
$$
$$
\mathcal L_{\text{grpo-clip-kl}}
=
\operatorname{Agg}\!\left(
-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{clip}}\right)
+\beta \widehat{\mathrm{KL}}_{i,t}
\right).
$$

---

### 3.6 dapo_clip

$$
\widetilde r_{i,t}^{\text{dapo}}
=
\operatorname{clip}(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}}).
$$
$$
\ell_{i,t}^{\text{dapo}}
=
-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{dapo}}\right),
$$
$$
\mathcal L_{\text{dapo}}
=
\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{dapo}}\right)\right).
$$

---

### 3.7 dapo_clip + overlong penalty

$$
r_i'
=
r_i-\lambda_{\text{len}}\frac{\max(L_i-L_0,0)}{L_0},
$$
$$
A_{i,t}^{\text{overlong}}=A_i^{\text{seq, overlong}}.
$$
$$
\ell_{i,t}^{\text{dapo-overlong}}
=
-\min\!\left(A_{i,t}^{\text{overlong}}r_{i,t},\; A_{i,t}^{\text{overlong}}\widetilde r_{i,t}^{\text{dapo}}\right),
$$
$$
\mathcal L_{\text{dapo-overlong}}
=
\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}^{\text{overlong}}r_{i,t},\; A_{i,t}^{\text{overlong}}\widetilde r_{i,t}^{\text{dapo}}\right)\right).
$$

---

### 3.8 dapo_clip + filter

$$
\mathcal L_{\text{dapo-filter}}
=
\operatorname{Agg}_{(x_i,y_i)\sim \mathcal D_{\text{kept}}}\!
\left(
-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{dapo}}\right)
\right).
$$

---

### 3.9 dapo full

$$
\mathcal L_{\text{dapo-full}}
=
\operatorname{Agg}_{(x_i,y_i)\sim \mathcal D_{\text{kept}}}\!
\left(
-\min\!\left(A_{i,t}^{\text{overlong}}r_{i,t},\; A_{i,t}^{\text{overlong}}\widetilde r_{i,t}^{\text{dapo}}\right)
\right).
$$

---

### 3.10 gbpo_clip

$$
\widetilde r_{i,t}^{\text{gbpo}}
=
\begin{cases}
\operatorname{clip}\!\left(r_{i,t},1-\epsilon_{\text{low}}^{+},1+\epsilon_{\text{high}}^{+}\right),
& A_{i,t}\ge 0,\\[6pt]
\operatorname{clip}\!\left(r_{i,t},1-\epsilon_{\text{low}}^{-},1+\epsilon_{\text{high}}^{-}\right),
& A_{i,t}<0.
\end{cases}
$$
$$
\ell_{i,t}^{\text{gbpo}}
=
-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{gbpo}}\right),
$$
$$
\mathcal L_{\text{gbpo}}
=
\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{gbpo}}\right)\right).
$$

---

### 3.11 gbpo_clip + KL

$$
\ell_{i,t}^{\text{gbpo-kl}}
=
-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{gbpo}}\right)
+\beta \widehat{\mathrm{KL}}_{i,t},
$$
$$
\mathcal L_{\text{gbpo-kl}}
=
\operatorname{Agg}\!\left(
-\min\!\left(A_{i,t}r_{i,t},\; A_{i,t}\widetilde r_{i,t}^{\text{gbpo}}\right)
+\beta \widehat{\mathrm{KL}}_{i,t}
\right).
$$

---

### 3.12 grpo_clip + step progress reward

$$
A_{i,t}^{\text{final}}
=
A_i^{\text{seq}}+\alpha A_{i,t}^{\text{step}}.
$$
$$
\ell_{i,t}^{\text{step+clip}}
=
-\min\!\left(A_{i,t}^{\text{final}}r_{i,t},\; A_{i,t}^{\text{final}}\widetilde r_{i,t}^{\text{clip}}\right),
$$
$$
\mathcal L_{\text{step+clip}}
=
\operatorname{Agg}\!\left(-\min\!\left(A_{i,t}^{\text{final}}r_{i,t},\; A_{i,t}^{\text{final}}\widetilde r_{i,t}^{\text{clip}}\right)\right).
$$

---

### 3.13 grpo_clip + step + KL

$$
\ell_{i,t}^{\text{step+clip+kl}}
=
-\min\!\left(A_{i,t}^{\text{final}}r_{i,t},\; A_{i,t}^{\text{final}}\widetilde r_{i,t}^{\text{clip}}\right)
+\beta \widehat{\mathrm{KL}}_{i,t},
$$
$$
\mathcal L_{\text{step+clip+kl}}
=
\operatorname{Agg}\!\left(
-\min\!\left(A_{i,t}^{\text{final}}r_{i,t},\; A_{i,t}^{\text{final}}\widetilde r_{i,t}^{\text{clip}}\right)
+\beta \widehat{\mathrm{KL}}_{i,t}
\right).
$$

---

### 3.14 OPSD

student 分布：
$$
p_\theta^s(\cdot\mid x_i,y_{i,<t}),
$$
teacher 分布：
$$
p_\theta^t(\cdot\mid x_i^{\text{teacher}},y_{i,<t}).
$$

token 级损失：
$$
\ell_{i,t}^{\text{opsd}}
=
k\cdot \operatorname{KL}\!\Big(
p_\theta^t(\cdot\mid x_i^{\text{teacher}},y_{i,<t})
\;\parallel\;
p_\theta^s(\cdot\mid x_i,y_{i,<t})
\Big).
$$

batch loss：
$$
\mathcal L_{\text{opsd}}
=
\frac{1}{|M|} \sum_{i=1}^{B} \sum_{t\in M_i}
k\cdot \operatorname{KL}\!\Big(
p_\theta^t(\cdot\mid x_i^{\text{teacher}},y_{i,<t})
\;\parallel\;
p_\theta^s(\cdot\mid x_i,y_{i,<t})
\Big).
$$

---

### 3.15 EI / SFT

$$
\ell_{i,t}^{\text{EI-SFT}}
=
- \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t}),
$$
$$
\mathcal L_{\text{EI-SFT}}
=
\frac{1}{|M|} \sum_{i=1}^{B} \sum_{t\in M_i}
- \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

