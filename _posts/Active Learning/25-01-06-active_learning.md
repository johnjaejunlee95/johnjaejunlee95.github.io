---
layout: post
gh-repo: johnjaejunlee95/johnjaejunlee95.github.io
gh-badge: [star, follow]
comments: true
author: johnjaejunlee95
title: "[개념리뷰] Elements of Information Theory"
date: "2025-06-29"
permalink: /information_theory/
description: ""
categories: [Information Theory, Entropy, Mutual Information]
toc: False
hits: true
# toc_sticky: True
tags: [Information Theory, Entropy, Mutual Information, KL Divergence, Jensen's Inequality, Data Processing Inequality]
use_math: true
author_profile: true
published: false
sidebar:
  nav: "docs"
---



<div>이번 포스트에서는 정보 이론의 핵심 개념인 Entropy, Relative Entropy, 그리고 Mutual Information에 대해 알아보겠습니다. 이 개념들은 데이터의 불확실성, 두 분포 간의 차이, 그리고 확률 변수들 간의 정보 공유량을 측정하는 데 사용됩니다.</div>

### Entropy (Self-Information)

**Entropy**는 확률 변수의 불확실성을 측정하며, 다음과 같이 정의됩니다:

$$
\begin{align*}
H(X) &= - \sum_{x \in \mathcal{X}}p(x) \log p(x) \\
&= \mathbb{E}_p \left[\frac{1}{\log p(X)}\right]
\end{align*}
$$

여기서 $X$는 이산 확률 변수이고, $p(x) = p_X(x) = \Pr\{ X=x\}$는 probability mass function입니다.

> **Remark 2.1:** Entropy는 확률 분포에 대한 것이며, 확률 변수 $X$가 취하는 실제 값(예: vectors)에는 의존하지 않습니다.

직관적으로 정보량(entropy)은 어떤 사건의 확률과 관련이 있습니다. 예를 들어, 어떤 일이 100% 확실하게 발생한다면 유용한 정보를 전혀 담고 있지 않으며, entropy는 0이 됩니다.

Entropy의 정의와 관련된 몇 가지 Lemma는 다음과 같습니다:

* **Lemma 1:** $H(X) \geq 0$
* **Lemma 2:** $H_b (X) = \log_b a \cdot H_a(X) = \log_b a \cdot \log_a p$

### Joint Entropy and Conditional Entropy

$H(X)$가 단일 확률 변수의 Entropy를 정의하듯이, 이를 한 쌍의 확률 변수 $(X,Y)$로 확장할 수 있으며, 이를 ***Joint Entropy***라고 합니다. 정의는 다음과 같습니다:

$$
\begin{align*}
H(X,Y) &= - \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log p(x,y) \\
&= \mathbb{E} \left[\frac{1}{\log p(X,Y)} \right]
\end{align*}
$$

또한, 쌍 $(X,Y)$에서 두 번째 변수에 조건부된 Entropy를 정의할 수 있는데, 이를 ***Conditional Entropy***라고 합니다. 정의는 다음과 같습니다:

$$
\begin{align*}
H(X\mid Y) &= \sum_{x\in\mathcal{X}} p(x) H(Y \mid X=x) \\
&= -\sum_{x\in\mathcal{X}}p(x) \sum_{y \in \mathcal{Y}} p(y \mid x) \log p(y \mid x) \\
&= -\sum_{x\in\mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log p(y \mid x) \\
& = \mathbb{E} \left[ \log \frac{1}{p(Y \mid X)} \right]
\end{align*}
$$

Joint Entropy와 Conditional Entropy의 정의를 바탕으로 다음 Theorem을 증명할 수 있습니다:

$$
\begin{align*}
H(X,Y) &= H(X) + H(Y \mid X) \\
\end{align*}
$$

***<highlights>Proofs:***

$$
\begin{align*}
H(X,Y)&= -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log p(x, y) \\
&= -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log p(x) p(y \mid x) \\
&= -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y)\log p(x) -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log p(y \mid x) \\
&= - \sum_{x \in \mathcal{X}} p(x) \log p(x) -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log p(y \mid x) \\
&= H(X) + H(Y \mid X)
\end{align*}
$$

### Relative Entropy and Mutual Information

***Relative Entropy***, $D(p \mid\mid q)$로 표기되며, 두 확률 분포 간의 차이를 측정합니다. 그러나 이는 진정한 distance metric이 **아닙니다**. **비대칭적**이며 **triangle inequality를 만족하지 않기** 때문입니다. 따라서 $D(p \mid\mid q)$는 두 분포 $p$와 $q$ 간의 "gap" 또는 divergence 측정으로 생각하는 것이 더 좋습니다.

이 측정 지표는 ***Kullback-Leibler Divergence (KL Divergence)***으로 알려져 있으며, 다음과 같이 정의됩니다:

$$
\begin{align*}
D(p \mid\mid q) &= \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}\\
&=\mathbb{E}_p \log \frac{p(X)}{q(X)}
\end{align*}
$$

KL Divergence는 항상 **non-negative(mostly positive)**이며, $p = q$인 경우에만 0과 같습니다.

> **Remark 2.2:** 딥러닝 분야에서 **분포 $\boldsymbol{p}$**는 종종 근사하려는 **true but unknown** distribution으로 가정되는 반면, **분포 $\boldsymbol{q}$**는 Gaussian Distribution과 같이 **known and tractable** distribution입니다. 이 개념은 KL Divergence를 최소화하여 모델이 실제 데이터 분포와 유사하도록 학습하는 Generative Models(예: VAE, Diffusion Models, Flow Matching)에서 널리 사용됩니다.

그런 다음 KL Divergence를 기반으로 한 확률 변수가 다른 확률 변수에 대해 포함하는 정보의 양을 측정할 수 있습니다. 이를 ***Mutual Information***이라고 하며, $\boldsymbol{I(X;Y)}$로 표기합니다. 이는 기본적으로 Joint Distribution과 marginal distributions의 곱 사이의 Relative Entropy이며, 다음과 같이 정의됩니다:

$$
\begin{align*}
I(X;Y) &= \sum_{x\in\mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}\\
&= D(p(x,y) \mid\mid p(x)p(y)) \\
&= \mathbb{E}_{p(x,y)} \left[\log \frac{p(X,Y)}{p(X)p(Y)} \right]
\end{align*}
$$

이 용어를 직관적으로 이해하기 위해서는 역으로 유도하는 것이 도움이 됩니다. 예를 들어, 먼저 $D(p(x,y) \mid\mid p(x)p(y))$의 의미를 고려해 봅시다. KL Divergence의 정의에 따르면, $D(p(x,y) \mid\mid p(x)p(y))$는 *how far the joint distribution is from independence* between $p(x)$ and $p(y)$를 나타냅니다. 만약 $p(x)$와 $p(y)$가 완전히 독립이라면, $p(x,y) = p(x)p(y)$가 되고, $D(p(x,y) \mid\mid p(x)p(y)) = 0$이 되어 $I(X;Y) = 0$이 됩니다. 이는 $X$와 $Y$ 사이에 공유되는 정보가 없음을 의미합니다.

### Relationship Between Entropy and Mutual Information

Mutual Information은 다음과 같이 Entropy 항으로 유도될 수 있습니다:

***Proofs:***

$$
\begin{align*}
I(X;Y) &= \sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}\\
&= \sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}} p(x,y) \log \frac{p(x \mid y) \textcolor{red}{\cancel{p(y)}}}{p(x)\textcolor{red}{\cancel{p(y)}}} \\
&= \sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}} p(x,y) \log \frac{p(x \mid y )}{p(x)} \\
&= - \sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}}p(x,y)\log p(x) + \sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}}p(x,y) \log p(x \mid y) \\
&= -\sum_{x \in \mathcal{X}}{{\sum_{y \in \mathcal{Y}} p(x,y)}}\log p(x) - \left(-\sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}} p(x,y) \log p (x \mid y) \right)\\
&= \underbrace{-\underset{x \in \mathcal{X}}{\sum} p(x)\log p(x)}_{H(X)} - H(X\mid Y) \\
&= H(X) - H(X \mid Y) \Box
\end{align*}
$$

이전에 설명했듯이, $I(\cdot; \cdot)$의 의미는 Entropy 개념을 통해 직관적으로 이해할 수 있습니다.

예를 들어, *transmitter*가 메시지($X$)를 전달하려고 한다고 가정해 봅시다. *receiver*는 $X$에 의해 영향을 받는 신호($Y$)를 관찰합니다. 이 맥락에서 Entropy $H(X)$는 전송되는 메시지의 총 불확실성, 즉 잠재적 정보량을 측정합니다. 수신기가 $Y$를 관찰하면 원래 메시지 $X$를 추론하려고 할 수 있습니다. 만약 $Y$가 $X$에 대한 완전한 정보를 제공한다면, $Y$를 관찰한 후의 $X$의 불확실성은 최소화됩니다. 만약 $Y$가 noisy하거나 불완전하다면, 불확실성은 더 높게 유지됩니다.

이렇게 남아있는 불확실성은 Conditional Entropy $H(X \mid Y)$로 포착됩니다. 다시 말해, $H(X \mid Y)$가 클수록 원래 메시지에 대한 불확실성이 더 많이 남아 있다는 것을 나타내며, 이는 수신기가 메시지를 완전히 복구하지 못했음을 의미합니다. $H(X \mid Y)$가 작을수록 수신기가 $X$를 더 정확하게 추론할 수 있었음을 의미합니다. 따라서 Mutual Information $\boldsymbol{I(X;Y)} = H(X) - H(X \mid Y)$는 수신기의 관찰($Y$)로 인해 송신자의 메시지($X$)에 대한 **불확실성의 감소량**을 정량화합니다.

다시 표기법으로 돌아가서, Entropy와 Mutual Information의 관계는 다음과 같이 요약할 수 있습니다:

1.  $I(X;Y) = H(X) - H(X \mid Y)$
2.  $I(X;Y) = H(Y) - H(Y \mid X)$
3.  $I(X;Y) = H(X) + H(Y) - H(X,Y)$
4.  $I(X;Y) = I(Y;X)$
5.  $I(X;X) = H(X)$

### Chain Rules of Entropy, Relative Entropy, and Mutual Information

Joint Entropy는 Conditional Entropy의 합으로 표현될 수 있습니다:

$$
\begin{equation*}
H(X_1, X_2, \ldots, X_n) = \sum_{n=1}^N H(X_n \mid X_{n-1}, \ldots, X_1)
\end{equation*}
$$

이 속성을 바탕으로 ***Conditional Mutual Information***을 다음과 같이 정의합니다:

$$
\begin{align*}
I(X;Y \mid Z) &= H(X \mid Z) - H(X \mid Y,Z) \\
&= \mathbb{E}_{p(x,y,z)} \left[\log \frac{p(X, Y \mid Z)}{p(X \mid Z)p(Y \mid Z)} \right]
\end{align*}
$$

이는 Mutual Information의 Chain Rule로 이어집니다:

$$
\begin{equation*}
I(X_1, \ldots, X_n \mid Y) = \sum_{n=1}^N I(X_n ; Y \mid X_{n-1}, \ldots, X_1)
\end{equation*}
$$

> **Remark 2.3:** 이러한 정의는 Probability Distributions의 근본적인 속성인 ***Chain Rule***에 기반을 두고 있습니다.

또한, ***Conditional KL Divergence***를 다음과 같이 정의할 수 있습니다:

$$
\begin{align*}
D (p(y \mid x) \mid\mid q(y \mid x)) &= \sum_{x \in \mathcal{X}}p(x) \sum_{y \in \mathcal{Y}} p(y \mid x)\log \frac{p(y \mid x)}{q(y \mid x)} \\
&= \mathbb{E}_{p(x,y)} \log \frac{p(Y \mid X)}{q(Y \mid X)}
\end{align*}
$$

### Jensen's Inequality and Its Consequences

정보 이론에서 Convex Functions의 속성은 근본적인 역할을 합니다. 함수는 다음과 같이 Convex Function으로 정의됩니다:

$$
\text{함수 $f(x)$가 구간 $(a,b)$에서 convex하다는 것은 모든 $x_1, x_2$와 $0 \leq \lambda \leq 1$에 대해 다음을 만족하는 것을 의미합니다.} \\
\begin{align*}
f(\lambda x_1 + (1-\lambda)x_2) \leq \lambda f(x_1) + (1-\lambda)f(x_2)
\end{align*}
$$

함수 $f$가 ***strictly convex***하다는 것은 등호가 $\lambda = 0$ 또는 $\lambda = 1$일 때만 성립하고, $-f$는 ***concave***하다고 합니다.

직관을 얻기 위해 Quadratic Function $f(x) = ax^2 + bx + c$를 생각해 봅시다. 두 번째 미분이 non-negative이면, 즉 $a \geq 0$이면, 함수는 convex합니다(또는 $a > 0$이면 strictly convex합니다). 더 일반적으로, Convexity는 점 $x_0$ 주변의 Taylor Series Expansion을 사용하여 분석할 수 있습니다:

***Proof:***

$$
\begin{align*}
f(x) &= f(x_0) + f^\prime (x_0)(x - x_0) + \frac{f^{\prime\prime}(x^\ast)}{2}(x - x_0)^2
\end{align*}
$$

여기서 $x^\ast$는 $x_0$와 $x$ 사이에 있습니다. $f^{\prime\prime}(x^\ast) \geq 0$이라고 가정하고, $x_0 = \lambda x_1 + (1-\lambda)x_2$로 놓습니다. $x = x_1$로 설정하면 다음을 얻습니다:

$$
\begin{align*}
f(x_1) &= f(x_0) + f^\prime(x_0)(x_1 - \lambda x_1 + (1 - \lambda)x_2) + \frac{f^{\prime\prime}(x^\ast)}{2}(x_1 - x_0)^2\\
&= f(x_0) + f^\prime (x_0)((1 - \lambda)(x_1 - x_2)) + \frac{f^{\prime\prime}(x^\ast)}{2}(x_1 - x_0)^2\\
&\geq f(x_0) + f^\prime (x_0)((1 - \lambda)(x_1 - x_2))
\end{align*}
$$

마찬가지로 $x = x_2$로 취하면:

$$
\begin{equation*}
f(x_2) \geq f(x_0) + f^\prime (x_0) (\lambda (x_2 - x_1))
\end{equation*}
$$

이제 $f(x_1)$에 $\lambda$를 곱하고 $f(x_2)$에 $(1 - \lambda)$를 곱한 다음 합하면:

$$
\begin{align*}
\lambda f(x_1) + (1 - \lambda) f(x_2) &\geq f(x_0) + \underbrace{\lambda f^\prime (x_0)((1 - \lambda)(x_1 - x_2)) + (1 - \lambda)f^\prime (x_0) (\lambda(x_2 - x_1))}_{\textcolor{red}{=-\lambda(1-\lambda)f^\prime (x_0)(x_2 - x_1) + \lambda (1 - \lambda) f^\prime (x_0)(x_2 - x_1) = 0}} \\
& = f(\lambda x_1 + (1 - \lambda)x_2)
\end{align*}
$$

이는 Convex Function의 정의를 복구하며, 증명을 완료합니다. $\Box$

---

다음으로, Machine Learning, Deep Learning, Information Theory 등 다양한 분야에서 가장 널리 사용되는 inequalities 중 하나인 ***Jensen's Inequality***를 소개합니다:

$$
\begin{equation*}
\mathbb{E}\left[ f(X) \right] \geq f(\mathbb{E}(X))
\end{equation*}
$$

여기서 $f$는 Convex Function이고 $X$는 Random Variable입니다.

Jensen's Inequality 덕분에 KL Divergence, Mutual Information 및 두 확률 분포 $p$와 $q$ 사이의 Conditional Forms와 같은 양은 Non-negative(대부분 Positive)임이 보장됩니다. Equality는 $p = q$인 경우에만 성립합니다.

### Log-Sum Inequality and Data-Processing Inequality

이제 Logarithm Function의 Concavity에서 비롯된 중요한 결과인 **Log-Sum Inequality**를 살펴보겠습니다. 이 Inequality는 Entropy가 Concave하다는 것을 증명하는 데 중요한 역할을 합니다:

$$
\begin{equation*}
\sum_{i=1}^n a_i \log \frac{a_i}{b_i} \geq \left( \sum_{i=1}^{n} a_i \right) \log \frac{\sum_{i=1}^n a_i}{\sum_{i=1}^n b_i}
\end{equation*}
$$

Equality는 모든 $i$에 대해 $\frac{a_i}{b_i} = C_i$ (constant)인 경우에만 성립합니다.

***==Proof:==***

$a_i > 0$이고 $b_i > 0$이라고 하고, 함수 $f(t) = t \log t$를 정의합니다. 이 함수는 $t > 0$일 때 $f^{\prime\prime}(t) = \frac{1}{t} \log e > 0$이므로 Strictly Convex합니다.

Jensen's Inequality를 적용하면:

$$
\begin{equation*}
\sum_{i=1}^n \alpha_i f(t_i) \geq f \left(\sum_{i=1}^n \alpha_i t_i \right)
\end{equation*}
$$

여기서 $\alpha_i \geq 0$이고 $\sum_i \alpha_i = 1$입니다. 이제 다음을 선택합니다:

$$\alpha_i = \frac{b_i}{\sum_j b_j}, \quad \text{and} \quad t_i = \frac{a_i}{b_i}.$$

그러면 Inequality는 다음과 같이 됩니다:

$$
\begin{equation*}
\sum_{i=1}^n \frac{a_i}{\sum_{j=1}^n b_j} \log \frac{a_i}{b_i} \geq \left( \sum_{i=1}^n \frac{a_i}{\sum_{j=1}^n b_j} \right) \log \left( \sum_{i=1}^n \frac{a_i}{\sum_{j=1}^n b_j} \right)
\end{equation*}
$$

양변에 $\sum_{j=1}^n b_j$를 곱하면 Log-Sum Inequality를 얻습니다. $\Box$

---

다음으로, **Data-Processing Inequality (DPI)**는 "no clever manipulation of data can increase information"이라는 직관을 공식화하는 Information Theory의 근본적인 결과입니다. 특히, Random Variable $X$가 Intermediate Variable $Y$를 통해서만 $Z$에 영향을 미친다면, $X$와 $Z$ 사이의 Mutual Information은 $X$와 $Y$ 사이의 Mutual Information을 초과할 수 없다고 명시합니다.

형식적으로 Markov Chain을 고려해 봅시다:

$$
\begin{equation*}
X \rightarrow Y \rightarrow Z
\end{equation*}
$$

이는 $X$와 $Z$가 $Y$가 주어졌을 때 conditionally independent임을 의미합니다. 즉,

$$
\begin{equation*}
p(z \mid x, y) = p(z \mid y).
\end{equation*}
$$

그러면 Markov Chain $X \rightarrow Y \rightarrow Z$에 대한 **Data-Processing Inequality**는 다음과 같이 명시합니다:

$$
\begin{equation*}
I(X; Z) \leq I(X; Y).
\end{equation*}
$$

다시 말해, Processing Data($Y \rightarrow Z$로부터)는 $Z$가 $X$에 대해 가지고 있는 정보의 양을 증가시킬 수 없습니다.

<br>

**$\*\$읽어주셔서 매우 감사합니다!! 혹시나 글을 읽으시다가 틀린 부분이 있거나 조언해주실 부분이 있다면 언제든 의견 전달주시면 감사하겠습니다!!** :smiley:
