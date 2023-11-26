# 机器学习实验2报告

> PB21000039 陈骆鑫

## 实验内容

使用两种不同的方法实现 SVM（支持向量机）算法，并对训练时间、准确率等参数进行比较。

## 实验原理

支持向量机是一种二分类算法。它的主要思路是找到一个超平面 $\pmb w ^T \pmb x + b = 0$，将不同类的样本分开，即满足：（样本量为 $N$）
$$
y_i (\pmb w ^T \pmb x_i + b) \ge 1, 1\le i \le N
$$

SVM 使用的最优化目标是“间隔”，即到样本点到平面间的最近距离。在使用前面的约束后，可以使用最大化目标 $2 / || \pmb w ||$，这与最小化 $\dfrac 1 2 || \pmb w || ^ 2$ 等价。

使用拉格朗日乘子法，可以得到该问题的对偶问题：

$$
\max _{\pmb \alpha} \sum_{i = 1} ^ {N} \alpha_i - \dfrac 1 2 \sum_{i=1}^{N} \sum_{j=1}^N \alpha_i \alpha_j y_i y_j \pmb x_i^T \pmb x_j\\
\mathrm{s.t.} \sum_{i=1}^N \alpha_i y_i = 0, \alpha_i \ge 0
$$

### 软间隔

前面的算法要求所有样本严格被划分超平面分开，而实际情况下，由于噪声等原因，这样的策略不一定能得到最优解——数据甚至不一定线性可分。针对这样的问题，可以使用“软间隔”的方法缓解。软间隔不要求，而使用惩罚函数惩罚不正确分类的样本。具体而言，最优化任务改写如下：

$$
\min _{\pmb w, b} \dfrac 1 2 || \pmb w|| ^ 2 + C \sum_{i=1}^N l(y_i(\pmb w ^T \pmb x_i + b) - 1)
$$

其中 $l$ 为选择的损失函数，$C$ 为选择的惩罚参数。若使用 hinge 损失函数 $l_hinge(z) = \max(0, 1-z)$，改写后的对偶问题如下：

$$
\max _{\pmb \alpha} \sum_{i = 1} ^ {N} \alpha_i - \dfrac 1 2 \sum_{i=1}^{N} \sum_{j=1}^N \alpha_i \alpha_j y_i y_j \pmb x_i^T \pmb x_j\\
\mathrm{s.t.} \ \sum_{i=1}^N \alpha_i y_i = 0, 0 \le \alpha_i \le C
$$


### SMO 算法

针对这个特定问题，可以使用 SMO 算法求解。该算法的基本思路是每次只取两个参数 $\alpha_i, \alpha_j$ 更新，而固定其余参数不变。而优化这两个参数的过程十分高效，有闭式解。具体而言，两个变量的问题如下：（参考《统计学习方法》）

$$
\max_{\alpha_1, \alpha_2} \dfrac 1 2 K_{11} \alpha_1^2 +  \dfrac 1 2 K_{22} \alpha_2^2 + y_1y_2K_{12} \alpha_1 \alpha_2
- (\alpha_1 + \alpha_2) + y_1 \alpha_1 \sum_{i=3}^N y_i \alpha_i K_{i1} + y_2 \alpha_2 \sum_{i=3}^N y_i \alpha_i K_{i2} \\
\mathrm{s.t.} \ \alpha_1 y_1 + \alpha_2 y_2 = - \sum_{i=3}^N \alpha_i y_i = \zeta, \\
0 \le \alpha_1, \alpha_2 \le C
$$

令：
$$
\left\{\begin{matrix}
L= \max(0, \alpha_2 - \alpha_1),& H  = \min(C, C + \alpha_2 - \alpha_1),& y_1 \ne y_2\\
L= \max(0, \alpha_2 + \alpha_1 - C),& H = \min(C, \alpha_2 +\alpha_1) ,& y_1 = y_2
\end{matrix}\right.
$$

则问题的未经剪辑的解是

$$
\alpha_2^{new,unc} = \alpha_2 + \dfrac{y_2(E_1 - E_2)}{\eta}
$$

，剪辑后的解是

$$\alpha_2^{new} = 
\left\{\begin{matrix}
H ,&\alpha_2^{new,unc} > H\\
\alpha_2^{new,unc}, & L \le \alpha_2^{new,unc} \le H, \\
L,& \alpha_2^{new,unc} < L

\end{matrix}\right. \\
$$
$$
\alpha_1^{new} = \alpha_1 + y_1 y_2(\alpha_2^{new} - \alpha_2)
$$



## 代码实现

下面解释核心代码的实现。

### SMO算法