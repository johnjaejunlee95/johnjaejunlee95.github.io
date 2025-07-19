---
layout: post
gh-repo: johnjaejunlee95/johnjaejunlee95.github.io
gh-badge: [star, follow]
comments: true
author: johnjaejunlee95
title: "[정보이론] Asymptotic Equipartition Property (AEP)"
date: "2025-07-19"
permalink: /info_aep/
description: ""
categories: [Information Theory]
toc: False
hits: true
# toc_sticky: True
tags: [entropy, AEP, WLLN, typical set, sequence length]
use_math: true
author_profile: true
published: true
sidebar:
  nav: "docs"

---

최근 AI를 공부하면서, 자연스럽게 **Information Theory(정보이론)**의 관점에서도 많은 생각을 하게 되었습니다. 그 과정에서 다시금 이론 자체를 깊이 들여다보게 되었고, 마침 블로그 콘텐츠가 한동안 뜸했던 참이라, 이를 계기로 "*Elements of Information Theory*" textbook 중심으로 공부한 내용을 정리해 포스팅해보면 좋겠다는 생각이 들었습니다. **직관적인 예시 중심의 설명**과 함께, 꼭 필요한 경우에는 **수식과 증명도 함께 곁들여** 최대한 이해하기 쉽게 정리해 나갈 예정입니다.

한가지 말씀드릴 점은 보통 정보이론을 다룰 때는 ***Chapter 2: Entropy, Relative Entropy, and Mutual Information***부터 시작하는 경우가 많지만, 해당 주제는 워낙 잘 정리된 자료들이 많기 때문에, 제가 공부하고 정리한 파일을 첨부하는 것으로 대체하고자 합니다. (파일은 바로 아래에) 그래서 정보이론 관련한 첫 posting은 정보 압축과 깊은 관련이 있는 **Chapter 3: Asymptotic Equipartition Property (AEP)**부터 시작하려고 합니다!!

<a href="https://raw.githubusercontent.com/johnjaejunlee95/johnjaejunlee95.github.io/main/assets/files/chapter_2_notes.pdf" download>Elementes of Information Theory - Chapter 2 Notes</a>



## Asymptotic Equipartition Property (AEP)

Asysmptotic Equipartition Property (AEP)란 용어가 어려워보이지만, 단순화 시키면 큰 수의 법칙(Law of Large Number)을 entropy 적용한 것으로 이해하면 됩니다. 그리고 이를 수학적으로 풀어낸걸로 이해하시면 됩니다. 그러면 formal한 definition부터 보면서 설명을 드리겠습니다.

임의의 random variable  $X_i \sim p(x)$로 부터 i.i.d sampling 및 이를 $N$번 시행한다고 가정해봅시다. 그러면 AEP의 정의를 다음과 같이 내릴 수 있습니다: 

### Theorem 1:

$$
\begin{align*}
-\frac{1}{N}\log p(X_1, X_2 , \ldots ,X_N) &= -\frac{1}{N}\sum_ {i=1}^N \log p(X_i) \\
&= - \mathbb{E}\left[ \log p(x) \right] \text{in probability} \\
 &\rightarrow H(X)
\end{align*}
$$

---

이게 어떤 의미인지를 살펴보겠습니다. 보통 entropy를 수식은 $- \sum_i p(X_i) \log p(X_i)$으로 정의가 됩니다. 그런데 현재 상황은 ***i.i.d***를 가정했기 때문에 $\log$ 앞에 붙은 $p(X)$ 를 고려할 필요가 없게됐습니다. 각 random variable이 sampling될 확률이 모두 동일하기 때문에 (=**독립시행**이기 때문에), 단순하게 $N$으로 나눠주면 되는 것입니다. 따라서 expectation도 $p(x)$를 고려한 notation $\mathbb{E}_p$가 아닌 $\mathbb{E}$를 취해준 것입니다. ***즉, AEP의 최종적인 의미는 i.i.d 상황에서 어떤 random variable의 sequence가 주어졌을 때, 이는 `entropy로 converge`된다는 의미입니다.***

> **Remark 3.1:** Machine Learning에서는 보통 expectation을 취한다는 것을 단순하게 평균을 구하는 것과 동일하게 취급합니다. 그 이유는 보통 data를 sampling 할 때 ***i.i.d 상황***을 가정하기 때문입니다.

사실, 이 부분에서 수학적으로 한가지 더 살펴볼 부분이 있습니다. 위 수식 중 $\frac{1}{N}\sum_i^N \log p(X_i)$의 경우 sample mean이고, $\mathbb{E}\left[ \log p(x) \right]$의 경우 true distribution $p(x)$의 전체 평균 값 (또는 expectation)입니다. 즉, 일부 sampling을 통해 계산한 일부 entropy value를 전체 entropy $H(X)$로 치환하기엔 수학적으로 완벽히 동일하다고 볼 순 없습니다. 하지만 성립할 수 있는 이유는 sample mean expectation이 true distribution의 expectation으로 converge한다라는 것이 이미 증명이 됐기 때문입니다. 그렇다면, 어떻게 증명을 할 수 있는지 한번 살펴보도록 하겠습니다. (초반에 AEP를 설명하는 부분에서 ***Law of Large Number***라는 용어가 등장했는데, 요 부분 증명 때문입니다.)



### Proof 1:

(증명 부분은 편의를 위해서 영어로 작성하도록 하겠습니다.)

Let $X_1, X_2, \ldots, X_N$ be a random sample of $N$ i.i.d. variables from the distribution $p(x)$. Moreover, the mean of the true distribution **(population mean)** $p(x)$ is defined as $\mathbb{E}[X] = \mu$, and its finite variance is denoted by $\text{Var}[X] = \sigma^2$. Let us first break down the proof for the expectation of the sample mean as follows:

Let the sample mean, denoted by $\bar{X}_ N$, be defined as:

$$
\begin{equation*}
\bar{X}_ {N} = \frac{1}{N} \sum_ {i=1}^N X_{i} 
\end{equation*}
$$

Then, the expectation of the sample mean is given by:

$$
\begin{align*}
\mathbb{E}\left[\bar{X}_ N \right] = \mathbb{E}\left[\frac{1}{N} \sum_ {i=1}^N X_i \right] = \frac{1}{N}\sum_ {i=1}^N \mathbb{E}\left[ X_i \right]
\end{align*}
$$

Here, since the expectation of each random variable $X_i$ is $\mu$, the expectation of the sample mean becomes:

$$
\begin{equation*}
\mathbb{E}\left[\bar{X}_ N \right] = \frac{1}{N}\sum_ {i=1}^N \mathbb{E}\left[ X_i \right] = \frac{1}{N} \times N \times \mu = \mu
\end{equation*}
$$

This shows that the expectation of the sample mean equals the true population mean (i.e., the sample mean is an unbiased estimator of the population mean).

Next, the proof of convergence of the sample mean to the true mean is twofold—based on the **Weak Law of Large Numbers (WLLN)** and the **Strong Law of Large Numbers (SLLN)**. In this case, we will focus only on **WLLN**, which is already sufficient for use in the context of the Asymptotic Equipartition Property (AEP).

##### Weak Law of Large Number (WLLN)

First, WLLN states that the sample mean converges in probability to the population mean. This means that for any small positive number $\epsilon$, the probability that the sample mean $\bar{X}_ N$ deviates from the true mean $\mu$ becomes arbitrarily small. This can be formally expressed as:

$$
\begin{equation*}
\lim_{n \rightarrow \infty} P(\left| \bar{X}_ N - \mu \right| \geq \epsilon ) = 0
\end{equation*}
$$

This can be proven using Chebyshev's Inequality, assuming finite variance $\sigma^2$. Let us first recall Chebyshev's Inequality, which states that for any random variable $Y$ with mean $\mathbb{E}\left[Y\right]$ and variance ${Var}\left[ Y \right]$:

$$
\begin{equation*}
P(\left| Y - \mathbb{E}\left[Y\right] \right| \geq \epsilon) \leq \frac{Var\left[ Y \right]}{\epsilon^2}
\end{equation*}
$$

Now, let $Y = \bar{X}_ N$, and we can derive the variance of the sample mean $\bar{X}_ N$ using properties of variance:

$$
\begin{align*}
Var \left[ \bar{X}_ N \right] &= Var\left[ \frac{1}{N}\sum_ {i=1}^N X_i \right] \\
&= \frac{1}{N^2} Var \left[ \sum_ {i=1}^N X_i \right] \\
&= \frac{1}{N^2} \sum_ {i=1}^N Var\left[X_i \right] \\
&= \frac{1}{N^2} \sum_ {i=1}^N \sigma^2 \;\; (Var[X_i] = \sigma^2 \; \text{for all i}) \\
&= \frac{\sigma^2}{N}
\end{align*}
$$

Since we already know $\mathbb{E}\left[\bar{X}_ N\right] = \mu$ and ${Var}\left[\bar{X}_ N \right] = \frac{\sigma^2}{N}$, we can apply **Chebyshev's Inequality** to the sample mean $\bar{X}_ N$:

$$
\begin{equation*}
P(\left| \bar{X}_ N - \mu \right| \geq \epsilon) \leq \frac{\sigma^2}{N \epsilon^2}
\end{equation*}
$$

As the number of samples $N \to \infty$, the right-hand side approaches 0:

$$
\begin{equation*}
\lim_{N \rightarrow \infty} P( \left| \bar{X}_ N - \mu \right| \geq \epsilon) = 0
\end{equation*}
$$

This tells us that the **probability of the sample mean deviating from the true mean** by more than $\epsilon$ becomes arbitrarily small as the sample size grows. Since probabilities are non-negative, this is equivalent to:

$$
\begin{equation*}
\lim_{N \rightarrow \infty} P(\left| \bar{X}_ N - \mu \right| < \epsilon) = 1 
\end{equation*}
$$

This completes the proof of convergence in probability of the sample mean to the population mean, as stated by the **WLLN**. $\Box$ 

---

<br>

## Typical Set

앞서서 열심히 AEP라는 개념을 봤는데요, 사실 이 개념이 나오게 된 이유는 **Typical Set**을 설명하기 위함입니다. Typical Set (정형적 집합)이란 어떤 긴 sequence가 가지고 있을 것으로 예상되는 대표적 특성을 의미합니다. 이렇게 얘기하면 조금 헷갈릴텐데, 동전던지기로 예를 보면서 한번 설명해보겠습니다.



### Example 1:

동전던지기를 총 100번 진행을 한다고 가정해봅시다. 그러면 동전 던지기는 모든 상황이 independent하기 때문에, 앞면(H)과 뒷면(T)으로 이루어진 $2^{100}$ 의 경우의 수가 나옵니다. 여기서 **Typical한 결과**와 **Non-typical한 결과**를 다음과 같이 설명할 수 있습니다.

   * **비정형적인(Non-typical) 결과**: 앞면 100번(H:100) 혹은 뒷면 100번(T:100) 만 나오는 경우. 정확히 앞면과 뒷면이 번갈아 50번씩 나오는 경우(HTHT...)
   * **정형적인(Typical) 결과**: 전체 결과가 (H:51, T:49) 등의 앞면과 뒷면이 나오는 비율이 반반에 가까운 경우

즉, 동전던지기의 경우 Typical Set이란 앞면과 뒷면의 비율이 반반에 가까운 결과들의 집합을 의미합니다.

---

좀 더 informal하게 설명하면, ***`Typical Set`***은 결국 ***`상식선에서 납득이 가능한 확률적 결과들의 집합`***이라고 표현할 수 있을 것 같습니다. 

그러면 이제 Typical Set에 대한 정의를 formal하게 표현해봅시다. Typical Set을 수식으로 표현하면 다음과 같이 표현할 수 있습니다:



### Definition 1:

Let $(x_1, \ldots x_n) \in \mathcal{X}^n$ denotes sequences with length $n$, and $\epsilon$ is very small positive number. Then typical set ${A}^{(n)}_\epsilon$ with respect to $p(x)$ is set of sequence  with the property

$$
\begin{align*}
2^{-n(H(X)+\epsilon)} \leq p(x_1, x_2, &\ldots, x_n)\leq 2^{-n(H(X)-\epsilon)} 
\end{align*}
$$

By applying logs and if $(x_1, \ldots x_n) \in A_\epsilon^{(n)}$, then

$$
\begin{align*}
{-n(H(X)+\epsilon)} &\leq \log p(X_1, X_2, \ldots, X_n)\leq {-n(H(X)-\epsilon)} \\
{H(X)-\epsilon} &\leq -\frac{1}{n}\log p(X_1, X_2, \ldots, X_n)\leq {H(X)+\epsilon} 
\end{align*}
$$

___

즉, probability of each random variable의 평균값 $\frac{1}{n}\sum_i \log p(X_i)$ 은 entropy $H(X)$ 및 $\epsilon$에 의해서 bound가 형성된다는 의미입니다. 이를 좀 더 정확하게 표현하자면 **$\epsilon$-Typical Set** 이라고 부릅니다.

그런데 여기서 잘 보면 가운데 $-\frac{1}{n}\sum_i \log p(X_i)$의 형태가 위에서 설명한 AEP와 밀접한 관계가 보입니다. 이는 **AEP**에 따라서 **typical set $A_\epsilon^{(n)}$의 properties를 정립**할 수 있습니다. 그렇다면 $A_\epsilon^{(n)}$에는 어떤 properties가 있는지 그리고 거기에 대한 proofs까지 살펴보도록 하겠습니다.



### Properties 1: 

1. When sequence length $n$ is sufficiently large, then $P(A_\epsilon^{(n)}) \geq 1 - \epsilon$
2. Let $\rvert A \rvert$ denotes the number of elements in the set $A$; $\rvert A_\epsilon^{(n)} \rvert\leq 2^{n(H(X) + \epsilon)}$
3. When sequence length $n$ is sufficiently large, then $\rvert A_\epsilon^{(n)} \rvert \geq (1-\epsilon)2^{n(H(X) - \epsilon)}$

---



### Proof 2:

##### Proof 2.1:

From the Asymptotic Equipartition Property (AEP), we know that

$$
\begin{equation*}
-\frac{1}{n}\sum_ {i=1}^n \log p(X_i) \rightarrow H(X)
\end{equation*}
$$

in probability. This implies that when $n$ is sufficiently large,

$$
\begin{align*}
p \underbrace{\left(\left|-\frac{1}{n}\log p(X_1, \ldots, X_n) - H(X) \right| \leq \epsilon \right)}_ {\in A_ \epsilon^{(n)}} \geq 1 - \delta,
\end{align*}
$$

where $\delta$ is a small positive number, consistent with the convergence in probability. In the case where $\delta = \epsilon$, this satisfies the first property. $\Box$

##### Proof 2.2:

We can derive a lower bound on the probability mass of the typical set based on its definition:

$$
\begin{align*}
1 &= \sum_ {\mathbf{x} \in \mathcal{X}^n} p(\mathbf{x}) \\
&\geq \sum_ {\mathbf{x} \in A^{(n)}_ \epsilon} p(\mathbf{x}) \\
&\geq \sum_ {\mathbf{x} \in A^{(n)}_ \epsilon} 2^{-n(H(X) +\epsilon)} \scriptsize{\text{(by the lower bound of the Typical Set definition)}}\\
&= 2^{-n(H(X) +\epsilon)} |A^{(n)}_ \epsilon|.
\end{align*}
$$

Multiplying both sides by $2^{n(H(X) +\epsilon)}$ yields $\rvert A^{(n)}_ \epsilon\rvert  \leq 2^{n(H(X) +\epsilon)},$ which satisfies the second property. $\Box$

##### Proof 2.3:

From the first property, we can derive an upper bound as follows:

$$
\begin{align*}
1 - \epsilon &\leq p(A_\epsilon^{n}) \\
&\leq \sum_ {\mathbf{x} \in A^{(n)}_ \epsilon} 2^{-n(H(X) -\epsilon)} \scriptsize{\text{(by the upper bound of the Typical Set definition)}}\\
&= 2^{-n(H(X) -\epsilon)} |A^{(n)}_ \epsilon|.
\end{align*}
$$

Multiplying both sides by $2^{n(H(X) - \epsilon)}$ gives  $\rvert A^{(n)}_ \epsilon\rvert  \geq 2^{n(H(X) - \epsilon)},$ which satisfies the third property. $\Box$

---

<br>

## Consequences of AEP: Data Compression

앞서서 **AEP**와 **$\epsilon$-Typical Set**에 대한 concept에 대해서 살펴봤습니다. 그러면 이러한 성질들을 가지고 어떤걸 할 수 있을까요? 몇가지가 있지만 그 중 제일 유용하게 사용되는 분야는 **Data Compression**입니다. 즉, AEP를 통해 data에 대해 무손실에 가깝게 압축이 가능해집니다. 어떻게 가능한지 이또한 예시를 통해 살펴보도록 하겠습니다. 



### Example 2:

이 역시 동전 던지기로 예시를 들어보겠습니다. 좀 더 극단적인 예시를 들기 위해서 확률을 동일하게 주는 것이 아니라 **앞면(H) 90%(P(H)=0.9)**, **뒷면(T) 10%(P(T)=0.1)**이라고 가정해 봅시다. 그랬을 때, 나올 수 있는 다음과 같습니다:

- $P(HHH) = 0.9 \times 0.9 \times 0.9 \approx 0.729$
- $P(H\times2, T) = 0.9 \times 0.9 \times 0.1 \approx 0.081$ (3가지 경우의 수)
- $P(H, T\times2) = 0.9 \times 0.1 \times 0.1 \approx 0.009$ (3가지 경우의 수)
- $P(TTT) = 0.1 \times 0.1 \times 0.1 \approx 0.001$

다음으로는 단순하게 생각했을 때 3번 동전을 던졌을 때 앞면(H)이 나올 예상 횟수는 $3 \times 0.9 = 2.7$로 볼 수 있습니다. 이 수치는 범위를 조금 넓게 잡아서 3번 던졌을 때 약 2번 또는 3번이 나오는 것으로 볼 수 있습니다 (lower bound = 2, upper bound = 3). 즉, 앞면(H)이 2번 또는 3번 나오는 것이 ***typical*** 하다고 할 수 있습니다. 이제 그러면 앞면(H)을 기준으로 $A^{(3)}$ 을 구하면 다음과 같습니다:

$A^{(3)} = P(HHH) + P(H\times2,T)\times3 = 0.729 +0.081 \times 3 = 0.972$

즉, 직관적으로 typical set을 찾았을 때, 해당 set이 발생할 확률은 약 97.2%로 볼 수 있습니다. 사실상 100%에 근접하죠.

다만 현재의 예시는 $n=3$으로 너무 작기 때문에 위에서 정의한 AEP 및 typical set에 대한 properties에 대입해서 보이기 어렵습니다. 왜냐하면 위에서의 properties는 $n$이 sufficently large일 때를 가정하기 떄문입니다. 따라서, 위에서의 직관을 기반으로, $n=100$까지 늘렸을 때 어떻게 나오는지 이번엔 수식적으로 살펴보겠습니다.

이제부터는 수식에서 entropy의 value를 직접적으로 활용해야하기 때문에 앞면이 나올 확률에 대한 entropy $H$를 구해보겠습니다. 앞면이 나올 확률에 대해 binary entropy $H(p)$ (Binomial trial)을 계산하면 다음과 같이 나옵니다:

$$
\begin{align*}
H(p) &= -p\log p - (1-p)\log (1-p) \scriptsize{\text{(Entropy for Binomial Trial)}}\\
H(0.9) &= 0.9 \log 0.9 - (1-0.9)\log 0.9 \\
&= 0.9 \log 0.9 - 0.1\log0.1 \\
&\approx 0.1368 + 0.3322\\
&\approx 0.469
\end{align*}
$$

즉, 한번 던질때마다 앞면(H) 기준 Entropy는 약 $\mathbf{H(0.9) = 0.469}$ 입니다. 그러면 이제 typical set $A_{\epsilon=0}^{(n)}$에 한번 대입을 해보면 다음과 같습니다: 

$$
\begin{equation*}
|A^{(n)}_{\epsilon=0}| = 2^{nH} = 2^{100\times0.469} = 2^{46.9} \approx 1.2\times10^{14}
\end{equation*}
$$

이 결과값의 의미를 정보이론 및 비트(bit) 관점에서 살펴봅시다. 

우선 위 예시에서 구한 것처럼 typical set이 발생할 확률을 먼저 구해봅시다. $n=3$일 때는 사실 bound를 잡기가 어려워 $\epsilon$을 따로 설정하지 않고 bound 범위를 단순하게 2 또는 3으로 지정했습니다. 그런데 이제 $n=100$이므로 $\epsilon$을 통해 typical set의 각 lower, upper bound를 설정할 수 있습니다. 이번 경우에서는 계산하기 편하게 예상 횟수를 $100 \times 0.9 = 90 =np$라고 하고, 여기에 bound를 $\pm 5$ 기준으로 typical set을 구해보겠습니다. (앞면 예상 횟수 범위: $N_H \in [85, 95]$)

$$
\begin{align*}
p(A^{(100)}_ \epsilon) &= \sum_ {k=85}^{95} p(N_H=k)\\
&=\sum_ {k=85}^{95}\binom{100}{k}p^k (1-p)^{100-k} \\
&\approx 0.936
\end{align*}
$$

즉, 전체 경우의 수 $2^{100}$ 중 typical set에 속하는 경우의 합이 전체 확률 질량의 약 **93.6%**를 차지한다는 의미입니다. 100%는 아니지만, 대부분의 경우가 typical set 안에 포함된다고 볼 수 있죠.

이제, 위에서 벌어진 일에 대한 정보를 누군가에게 전달해본다고 생각을 해봅시다. 동전 던지기에 대한 정보를 전달할 수 있는 총 가짓수는 $2^{100}$ (총 set $\mathcal{X}$ 의 size; $\rvert \mathcal{X}\rvert $ ) 입니다. 이를 bit 단위로 전송한다고 생각을 해보면 $\log2^{100} = 100$ bit가 필요할 것입니다(prefix 제외). 그런데 위에서 구한 typical set의 size는 약 $\log2^{46.9} = 46.9$ bit $ \approx 47$ bit입니다. `(이를 indexing 한다고 표현)` Typical Set만을 전송한다면 약 $53$ bit를 줄일 수 있게 된 것입니다. 이는 통신 관점에서는 매우 많은 양이 압축된다고 볼 수 있습니다. ($53\%$ 압축 가능)

이제, 위에서 벌어진 일에 대한 정보를 누군가에게 전달한다고 생각해봅시다. 동전 던지기에 대한 정보를 전달할 수 있는 총 가짓수는 $2^{100}$ (전체 set $\mathcal{X}$의 size)이고, 이걸 bit 단위로 전송한다고 하면 $\log_2 2^{100} =$ 100 bit가 필요합니다. 그런데 우리가 typical set만 전달한다고 하면, $2^{46.9} \approx 1.2 \times 10^{14}$ 개의 sequence만 구분하면 되므로, 약 $\log_2 2^{46.9} = $ 46.9 bit, 즉 **약 47bit**면 충분합니다. 결과적으로 **약 53bit를 줄일 수 있는 셈**이고, 이는 **전체 정보량의 절반 이상을 압축**할 수 있다는 뜻입니다. 

정리를 하자면, 동전 던지기 100회를 실시한 사건에 대한 정보를 누군가와 통신한다고 했을 때, typical set을 기준으로 보면 **전체 확률 질량의 약 93.6%에 해당하는 사건들**을 **약 47bit**, 즉 **기존보다 53% 더 적은 비트 수로 압축**하여 통신할 수 있다는 의미가 됩니다. 



`한줄요약: 자주 발생하는 set은 압축을 하고, 잘 발생하지 않는 set은 그대로 두자!`

---



이제 이를 sequence length 관점에서 formal하게 도록 하겠습니다. 우선, 앞서 정의한 typical set $A^{(n)}_ \epsilon$은 다음의 성질을 만족합니다:

$$
\begin{equation*}
|A^{(n)}_ \epsilon| \leq 2^{n(H + \epsilon)}
\end{equation*}
$$

즉, 대부분의 sequence들은 이 typical set 안에 포함되어 있고, 그 개수는 최대 $2^{n(H+\epsilon)}$개입니다. 예시에서 설명한 것처럼 이 typical set 안의 sequence들은 **약 $n(H + \epsilon) + 1$ bit 내에서 표현 가능**하다는 뜻이 됩니다. `(여기서 +1의 의미는 혹시` $n(H + \epsilon)$`가 integer가 아닐 때를 가정해 설정)` 여기에 추가로 typical set이냐 아니냐에 대한 정보도 필요하기 때문에 1 bit가 더 늘어나서 최종적으로 $A^{(n)}_ \epsilon$의 total length는 $n(H + \epsilon) +2$ bit 내에서 표현이 가능합니다. 이제 여기서 또 살펴볼 지점이 있는데, 위에서 설명한대로 typical set은 압축을 하고(indexing) non-typical set은 그대로 둔다고 했는데, 각각 total length를 다음과 같이 표현할 수 있습니다:

- Typical Set: $n\log \rvert A^{(n)}_ \epsilon\rvert  \leq n(H + \epsilon) +2 $ 
- Non-typical Set: $n\log \rvert \mathcal{X}\rvert  \leq n\log \rvert \mathcal{X}\rvert  + 2$ 

이제 sequence length의 expected value 구할 수 있습니다. 우선 $x^n=$ sequence $(x_1, \ldots x_n)$,  그리고 $x^n$에 대한 length를 $l(x^n)$이라 나타내겠습니다. 그러면 필요한 sequence length에 대한 expected value는 expectation을 취함으로써 다음과 같이 나타낼 수 있습니다:

$$
\begin{align*}
\mathbb{E}\left[ l(X^n) \right] &= \sum_ {x^n} p(x^n)l(x^n)\\
&= \sum_ {x^n \in A^{(n)}_ \epsilon } p(x^n)l(x^n) + \sum_ {x^n \in A^{(n)^c}_ \epsilon} p(x^n)l(x^n)\\
&\leq \sum_ {x^n \in A^{(n)}_ \epsilon } p(x^n)\left(n(H + \epsilon) + 2\right) + \sum_ {x^n \in A^{(n)^c}_ \epsilon} p(x^n)\left(n\log |\mathcal{X}| + 2 \right)\\
&= p(A^{(n)}_ \epsilon )\left(n(H + \epsilon) + 2\right) + p(A^{(n)^c}_ \epsilon )\left(n\log |\mathcal{X}| + 2 \right)\\
\end{align*}
$$

이제 여기서 또 고려해야할 부분은, 전체 sequence $X^n$ 에 대해 몇 비트를 쓰는지뿐만 아니라, **각 $x_i$ 당 몇 bit**가 필요한지를 알고 싶기 때문입니다. 즉, 전체 비트 수 $\mathbb{E}[l(X^n)]$가 아닌 평균 비트 수 (bit rate) $\frac{1}{n} \mathbb{E}[l(X^n)]$가 필요합니다. 관련해서 다음과 같이 정리할 수 있습니다:



### Theorem 2

Let $X^n \sim p(x)$. Let $\epsilon > 0$. Then there exists sequences $x^n$ with sufficently large length $n$ as follows:

$$
\begin{equation*}
\mathbb{E}\left[\frac{1}{n}l(X^n) \right] \leq H(X) + \epsilon
\end{equation*}
$$

---

이 Theorem은 바로 위에서 구한 expectation으로부터 구할 수 있습니다. 우선  $n\rightarrow \infty$일 때   $p(A^{(n)}_ \epsilon)$와 $p(A^{(n)^c}_ \epsilon)$ 은 Property 1.1에서 정리된 것을 보면 $p(A^{(n)}_ \epsilon) \rightarrow 1$, $p(A^{(n)^c}_ \epsilon) \rightarrow 0$으로 수렴하게 됩니다. 그러면 $\mathbb{E}\left[ l(X^n) \right] \leq n(H + \epsilon) + 2$이 되고 양변을 $n$으로 나누면 `Theorem 2` 와 같이 도출됩니다. 

결론적으로는 $n\rightarrow\infty$ 일 때 전체 sequence $X^n$를 약 (weakly) $nH$ bit 로 sequence 압축할 수 있다는 의미입니다.

<br>

## Conclusion

다뤘던 내용들에 대해서 간략히 정리하자면:



#### AEP (`Theorem 1`):

$$
\begin{equation*}
-\frac{1}{N}\sum_ {i=1}^N \log p(X_i) \rightarrow H(X)
\end{equation*}
$$

---

#### Typical Set (`Definition 1`, `Properties 1`):

**`Definition 1:`**
$$
\begin{equation*}
{H(X)-\epsilon} \leq -\frac{1}{n} \log p(X_1, X_2, \ldots, X_n)\leq {H(X)+\epsilon}
\end{equation*}
$$

**`Properties 1`**

1. When sequence length $n$ is sufficiently large, then $P(A_\epsilon^{(n)}) \geq 1 - \epsilon$
2. Let $\rvert A\rvert  $ denotes the number of elements in the set $A$; $\rvert A_\epsilon^{(n)}\rvert  \leq 2^{n(H(X) + \epsilon)}$
3. When sequence length $n$ is sufficiently large, then $\rvert A_\epsilon^{(n)}\rvert  \geq (1-\epsilon)2^{n(H(X) - \epsilon)}$

---

#### Convergence of Sequence Length (`Theorem 2`)

$$
\begin{equation*}
\mathbb{E}\left[\frac{1}{n} l(X^n) \right] \leq H(X) + \epsilon
\end{equation*}
$$

---

<br>

이번 chapter에서는 정보를 어떻게 효율적으로 압축하고 전달할 수 있는지를, 직관적인 설명과 함께 수학적으로도 어떻게 가능한지를 살펴보았습니다. 아마 앞으로도 ***Elements of Information Theory***에서 핵심이 되는 개념들 중, 특히 **AI 연구 관점에서 중요하게 작용하는 내용들**을 중심으로 정리해 나갈 예정입니다.

그럼 다음 포스팅에서 다시 찾아뵙겠습니다!

