# 极限

## 极限定义

$$
当x\to \infty时函数极限定义\\
\lim\limits_{x\to\infty}f(x) = A \Leftrightarrow \forall \varepsilon > 0,\exists X>0,当\,|x|>X时,有|f(x) - A|< \varepsilon.
$$

<br>
$$
当x\to x_0时函数极限定义\\
\lim\limits_{x\to x_0}f(x) = A \Leftrightarrow \forall \varepsilon > 0,\exists \delta > 0,当\,0<|x-x_0|< \delta 时,有|f(x) - A| < \varepsilon.
$$
<br>

## 极限存在

$$
\lim_\limits{x\to x_0}f(x) = A \Leftrightarrow \lim_\limits{x\to x^+_0}f(x) =  \lim_\limits{x\to x^-_0}f(x)=A
\\该条件说明了某点处是否有极限与该点是否有定义、该点的值是否等于极限无关
$$

<br>

## 连续定义

$$
\lim_\limits{x\to x_0}f(x) = f(x_0)
\\该公式为极限存在的拓展,表达了左右极限等于该点函数值的含义
$$

<br>

## 技巧

### 需要讨论左右极限的函数

- 某些函数在趋向某个值的时候左右极限不一样，遇到这种情况需要分情况讨论
- 注意这些函数复合的时候要小心

| 趋向          | 函数                                                         |
| ------------- | ------------------------------------------------------------ |
| $x\to \infty$ | $e^x$<br /><br />$\arctan x$<br /><br />$arc\cot x$<br /><br />$\frac{\sqrt{x^2+1}}{x}$ |
| $x\to 0$      | $\frac{1}{x}$<br /><br />$\frac{|x|}{x}$                     |
| $x\to a$      | $[x]$ *(向x轴负方向取整)*                                    |

<br>

### 从极限四则运算中学到什么

- 若 $\lim_\limits{x\to 0}\frac{f(x)}{g(x)} = A \neq 0$ 且 $\lim_\limits{x\to 0}f(x) = 0$，则$\lim_\limits{x\to 0}g(x) = 0$

- 七种未定式
  - 未定式诞生于极限四则运算中结果的不确定
    - $!\exists \,\,\pm\,\, !\exists = ？$
    - $!\exists \,\,*\,\, \exists = ？$
    - $!\exists \,\,*\,\, !\exists = ？$
  - 由此可知
    - 加减因子极限存在先算
    - 乘除非零因子极限存在先算

| 未定式                  | 解决方法                                                     |
| ----------------------- | ------------------------------------------------------------ |
| $\frac{\infty}{\infty}$ | 分子分母同除以最高项数；<br /><br />洛必达；                 |
| $\frac{0}{0}$           | 洛必达；<br /><br />等价无穷小；                             |
| $0·\infty$              | 转化为$\frac{0}{0}$或$\frac{\infty}{\infty}$；               |
| $\infty - \infty$       | 平方差公式创造分母；<br /><br />通分；<br /><br />倒代换；   |
| $f(x)^{g(x)}$           | 若满足$1^{\infty}$，$e^{\lim g(x)[f(x) - 1]}$；<br /><br />否则，$e^{\lim g(x)\ln f(x)}$；<br /><br />泰勒，万物皆可指数化；<br /><br />幂指函数相减，$e^A-e^B=(A-B)e^B$ |

<br>

#### 洛必达

使用洛必达的前提条件：$\frac{\infty }{\infty }$、$\frac{0}{0}$

<br>

#### 其他技巧

| 情况       | 解决方法                                                     |
| ---------- | ------------------------------------------------------------ |
| 反三角函数 | 用反三角函数的反函数换元                                     |
| 特殊情况   | $xf(x) = \frac{f(x)}{\frac{1}{x}}$<br /><br />$x$是$\ln x$的高阶无穷小<br /><br />$\lim_\limits{x\to 0^+}x^a(\ln x^b)=0,(a>0, b>0)$ |

<br>

## 性质

### 唯一性

- 若极限存在则极限唯一

### 局部有界性

- 若极限在某区间内存在，则该区间内有界
- 有如下推论

$$
若f(x)在(a,b)上连续,\lim_\limits{x\to a^+}f(x)和\lim_\limits{x\to b^-}f(x)都存在,则
\\f(x)在(a,b)内有界
$$

### 局部保号性

$$
若\lim_\limits{x\to x_0}f(x)=A,则在极限所在的去心领域内必然存在f(x) \ge 0(\le 0),使得A\ge 0(\le 0)
$$

