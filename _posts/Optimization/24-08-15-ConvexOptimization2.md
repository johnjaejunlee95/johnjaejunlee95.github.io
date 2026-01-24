---
title: "[Concept] Convex Optimization 2"
tags: [Convex, Optimization, Gradient Descent]
mathjax: true
show_subscribe: false
license: false
show_author_profile: true
---

<div>I’m back with the second post related to optimization. Last time, we examined how it expands and converges when the function is L-lipschitz. This time, let's examine what happens when stronger constraints / assumptions are applied. Then, without further ado, let's jump right in. </div>

### $\star$ Recap [(Previous Post)](https://johnjaejunlee95.github.io/Optimization_1/)

In the previous post, assuming a function is <mark style="background: orange">convex</mark> + $L$-lipschitz, the following Theorem was derived:

<mark style="background:skyblue" >Theorem</mark> Let $f$ be convex and $L$-Lipschitz continuous. Then gradient descent with $\gamma = \frac{\mid\mid x\_1 - x^\star\mid\mid}{L\sqrt{T}}$  satisfies:

$$f \left ( \frac{1}{T} \sum_{k=1}^T x_k  \right) - f(x^{\star}) \leq \frac{\mid\mid x_1 - x^\star \mid\mid L}{\sqrt{T}} \Rightarrow \mathcal{O}(\frac{1}{\sqrt{T}})$$

Then, what statement would emerge if we apply a stronger constraint or assumption than $L$-Lipschitz? Intuitively, it will probably converge at a much faster speed. Then let's verify through proof if it actually converges at a faster speed. (All examples are Gradient Descent.)



### Before we begin...

When dealing with optimization problems, assumptions/constraints can be applied in various ways. (From now on, I will express all situations where assumptions/limits are set as ***assumption***.) The stronger the assumption, the fewer functions satisfy it, but the convergence speed becomes faster. Therefore, by appropriately placing these constraints, we can verify "why it works well" from a Learning Theory perspective. The first assumption we looked at was $L$-Lipschitz, and this time we will look at how gradient descent converges under the $\beta$-Smoothness condition.

## Problem Formulation 2 <br> (Assumption: $\beta$ -Smooth)

Basically, the definition of $\beta$-Smoothness is as follows:

<mark style="background:lightgreen" >Definition</mark> A continuously differentiable function $f$ is $\beta$-smooth if the gradient $\nabla f$ is $\beta$-Lipschitz:


$$
||  \nabla f(x) - \nabla f(y) ||  \leq \beta || x - y ||
$$


In other words, if an arbitrary function $f$ is in a differentiable form, it means it is $\beta$-Lipschitz with respect to the differentiated function. If you think about why this is a stronger assumption, you can think of the concept of differentiation we learned in high school.

In the case of $L$-Lipschitz, it means that the rate of change for an arbitrary function $f$ is limited in proportion to $L$. However, $\beta$-Smooth means that the rate of change of the derivative value of an arbitrary function $f$ is limited in proportion to $\beta$. That is, $L$-Lipschitz means function $f$'s rate of change $\rightarrow$ derivative value is proportional to $L$, and $\beta$-Smoothness means function $f$'s derivative value's rate of change $\rightarrow$ second derivative value is proportional to $\beta$. In this case, all $\beta$-Smooth belong to $L$-Lipschitz, but since not all $L$-Lipschitz belong to $\beta$-Smooth, **$\beta$-Smooth can be seen as a stronger assumption.**

Returning to the main point, let's check how gradient descent converges when an arbitrary function $f$ is $\beta$-smooth.

<mark style="background:skyblue" >Theorem 2:</mark> Let $f$ be convex and $\beta$-smooth on $\mathbb{R}^n$ \. Then, gradient descent with $\gamma = \frac{1}{\beta}$ satisfies


$$
f(x_T) - f(x^\star) \leq \frac{2\beta || x_1 - x^\star || ^2}{T-1}
$$


The expansion method is similar to when it was $L$-Lipschitz. However, since a stronger assumption is included, we just need to look at a few more things.

### Proof:

Before we begin, there is one more definition we can define due to the property of $\beta$-smooth:


$$
f(y)  \leq f(x) + \left< \nabla f(x) , y-x\right> + \frac{\beta}{2} || y - x ||^2
$$


Since there is already an assumption that function $f$ is twice-differentiable, this formula can be expanded using the concept of Taylor Expansion. (Since this is not the main point but utilized as a definition, I will leave the proof as a [link](https://angms.science/doc/CVX/CVX_betasmoothsandwich.pdf).)

Then now, let's proceed with the proof for the convergence of gradient descent when it is really $\beta$-Smoothness. The direction of expansion itself is similar to when it is $L$-Lipschitz. However, you can see that the definitions available to use have increased. First, the gradient descent equation based on the property of $\beta$-Smooth can be expanded as follows.


$$
\begin{align}
  f(x_{t+1}) - f(x_t) &\leq \left< \nabla f(x_{t}) , x_{t+1} - x_t \right> + \frac{\beta}{2} || x_{t+1} - x_t||^2 \\
  &= \left< \nabla f(x_t) , -\gamma\nabla f(x_t) \right> + \frac{\beta}{2} ||-\gamma\nabla f(x_t)||^2 \\
  &= -\gamma ||\nabla f(x_t)||^2 + \frac{\beta}{2}*\gamma^2 ||\nabla f(x_t)||^2\\
  &= -\frac{1}{\beta} ||\nabla f(x_t)||^2 + \frac{1}{2\beta}||\nabla f(x_t)||^2 \;\; \text{where } \gamma = \frac{1}{\beta}\\
  &= -\frac{1}{2\beta}||\nabla f(x_t)||^2
  
\end{align}
$$



Also, like the proof when it was $L$-Lipschitz, the bound of $f(x_t) - f(x^\star)$ can be expanded as follows using the convexity of $f$: $f(x_t) - f(x^\star) \leq \left< \nabla f(x_t) , x_t - x^\star \right>$

The term we can utilize here is $\nabla f(x_t)$ when thinking of the proof above. In other words, since the bound related to $\mid\mid \nabla f(x_{t})\mid\mid^2$, which is the squared form of $\nabla f(x_t)$, is already available, utilizing this allows us to expand the equation a bit more smoothly. This can be done by utilizing the **Cauchy-Schwarz Inequality** property.

 ($\star$ **Cauchy-Schwarz Inequality**: $\left< a,b \right> \leq \mid\mid a \mid\mid \cdot \mid\mid b \mid\mid $)


$$
\begin{align}
\left[f(x_t) - f(x^\star)\right]^2 &\leq \left| \left< \nabla f(x_t) , x_t - x^\star \right>\right|^2 \\
&\leq || \nabla f(x_t) ||^2 \cdot || x_t - x^\star ||^2 \\
\Rightarrow \frac{\left[f(x_t) - f(x^\star)\right]^2}{ || x_t - x^\star ||^2} &\leq || \nabla f(x_t) ||^2 
\end{align}
$$


Now, for the next step, we need to re-arrange it into a form that is easy for us to handle, and the easiest way is ultimately to handle the bound for $f(x_{t+1}) - f(x_t)$. By subtracting $f(x^\star)$ from both inequality terms as follows, we can utilize all the methods expanded previously.

However, before that, there is one part to check. Before looking at $f$, we need to check how the bound is formed according to step $t$. Therefore, before looking at $f(x_{t+1}) - f(x_t)$, let's check how the bound between $x_t$ and $x_{t+1}$ is formed.


$$
\begin{align}
    || x_{t+1} - x^\star ||^2 &= || x_t - \frac{1}{\beta} \nabla f(x_t) - x^\star ||^2 \\
    &= ||x_t -x^\star||^2 - \frac{2}{\beta} \underbrace{\left< \nabla f(x_t), x_t - x^\star \right>}_{\leq \frac{1}{\beta} ||\nabla f(x_t)||^2} +\frac{1}{\beta^2} ||\nabla f(x_t)||^2 \\
    &\leq ||x_t - x^\star||^2 \underbrace{- \frac{1}{\beta^2} ||\nabla f(x_t)||^2}_{\text{always decreases}} \\
    &\leq ||x_t - x^\star||^2 \underbrace{- \frac{1}{\beta^2}\frac{\left[f(x_t) - f(x^\star)\right]^2}{ || x_t - x^\star ||^2}}_{\text{smaller decreases}}
\end{align}
$$

Utilizing this property, we can continue the proof using $f(x^\star)$ as follows.


$$
\begin{align}
    f(x_{t+1}) &\leq f(x_t) - \frac{1}{2\beta} ||\nabla f(x_t)||^2 \\
    f(x_{t+1}) - f(x^\star) = &\leq f(x_t) - f(x^\star) - \frac{1}{2\beta}||\nabla f(x_t)||^2 \\
    D_{t+1} &\leq D_t  - \frac{1}{2\beta}||\nabla f(x_t)||^2 \quad \text{where } D_{t+1} = f(x_{t+1}) - f(x^\star)\\
    D_{t+1} &\leq D_t - \frac{1}{2\beta}\cdot \frac{\left[f(x_t) - f(x^\star)\right]^2}{||x_t - x^\star ||^2} \\
    D_{t+1} &\leq D_t - \frac{1}{2\beta} \cdot \frac{D_t^2}{||x_t - x^\star ||^2}
\end{align}
$$



If we divide the inequalities above by $D_t D_{T+1}$


$$
\begin{align}
    \frac{1}{D_{t}} - \frac{1}{D_{t+1}} &\leq - \frac{1}{2\beta||x_t - x^\star ||^2}\frac{D_t}{  D_{t+1}}
\end{align}
$$


It becomes. However, here we can reset the bound through 2 conditions.

1. Since ${D_t}/{D_{t+1}}  = \left[f(x_t) - f(x^\star)\right] / \left[f(x_{t+1}) - f(x^\star)\right]$  and $f(x_{t+1})$ is smaller than $f(x_t)$, we know that ${D_t}/{D_{t+1}} \geq 1$.
2. Since ${x_t - x^\star} \leq x_1 - x^\star $, we know that $1 /{\mid\mid x_t - x^\star \mid\mid^2} \geq 1 / {\mid\mid x_1 - x^\star \mid\mid^2}$.

And since the Right Hand Side (RHS) term is a negative form, if we hold the bound tighter, it can be expressed as follows.


$$
\frac{1}{D_{t+1}} - \frac{1}{D_{t}} \geq \frac{1}{2\beta||x_1 - x^\star ||^2}
$$


Just like the previous expansion, if we substitute $t=1, \ldots , {T-1}$ sequentially


$$
\begin{align}
\frac{1}{D_{2}} - \frac{1}{D_{1}} &\geq \frac{1}{2\beta||x_1 - x^\star ||^2} \\
\frac{1}{D_{3}} - \frac{1}{D_{2}} &\geq \frac{1}{2\beta||x_1 - x^\star ||^2} \\
    &\;\;\vdots \\
\frac{1}{D_{T}} - \frac{1}{D_{T-1}} &\geq \frac{1}{2\beta||x_1 - x^\star ||^2} \\
\end{align}
$$


And if we proceed with summation for the inequalities above



$$
\begin{align}
    \sum_{t=1}^{T-1} \left[\frac{1}{D_{t+1}} - \frac{1}{D_{t}} \right] &\geq \sum_{t=1}^{T-1} \left[\frac{1}{2\beta ||x_1 - x^\star ||^2}\right] \\
    \frac{1}{D_T} - \frac{1}{D_1} &\geq \frac{T-1}{2\beta ||x_1 - x^\star ||^2} \\
    \frac{1}{D_T} &\geq \frac{1}{D_1} + \frac{T-1}{2\beta ||x_1 - x^\star ||^2}
    \end{align}
$$


Here, utilizing the property of $\beta$-Smooth and properties of convexity, we can set $D_1$ as follows.



$$
\begin{align}
    D_1 = f(x_1) - f(x^\star) &\leq \left< \nabla f(x^\star), x_1 - x^\star \right> + \frac{\beta}{2} ||x_1 - x^\star ||^2\\
    &\leq \frac{\beta}{2} ||x_1 - x^\star ||^2

\end{align}
$$

The reason this is possible is that ultimately, since $x^\star$ is a minimizer value, we can assume the derivative value for $f$ converges to 0. Therefore, since $\nabla f(x^\star) = 0$, we can derive the equation above.



And if we organize the formula


$$
\begin{align}
    \frac{1}{D_T} &\geq \frac{1}{D_1} + \frac{T-1}{2\beta ||x_1 - x^\star ||^2}\\
    \frac{1}{f(x_T) - f(x^\star)} &\geq \frac{1}{f(x_1) - f(x^\star)} + \frac{T-1}{2\beta ||x_1 - x^\star ||^2} \\
    \frac{1}{f(x_T) - f(x^\star)} &\geq \frac{2}{\beta ||x_1 - x^\star ||^2} + \frac{T-1}{2\beta ||x_1 - x^\star ||^2} \\
    \frac{1}{f(x_T) - f(x^\star)} &\geq \frac{T + 3}{2\beta||x_1 - x^\star ||^2}\\
    &\geq \frac{T -1}{2\beta||x_1 - x^\star ||^2}
\end{align}
$$


The reason for changing the numerator from $T-3 \Rightarrow T-1$ above is that since we checked from step $1$ to $T-1$ when we did the summation, matching the form makes it easier to set the bound.

Now, for the final step, if we take the reciprocal for each term, it's done. **(inequality direction changes)**


$$
\begin{align}
    f(x_T) - f(x^\star) \leq \frac{2\beta || x_1 - x^\star || ^2}{T-1} \Rightarrow \mathcal{O} \left(\frac{\beta}{T}\right)
    \end{align}
$$


## Afterwards...

In expanding the formulas, I tried to explain as best as I could, but as the conditions to satisfy increased, the explanation seems to have become a bit verbose. Because of that, you might feel that readability and flow are not smooth. I will try to polish it little by little whenever I have time.

I will stop the proofs related to Convex here. (Most are proved similarly.) I will return with different content in the next post.



<br>

**$\*\$Thank you very much for reading. If there are any incorrect parts while reading or if you have any advice, I would appreciate it if you could share your opinions anytime.**