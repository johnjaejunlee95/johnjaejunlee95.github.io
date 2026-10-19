---
layout: distill
title: "[Paper Review] Sharp-MAML"
date: 2023-03-13 00:00:00 +0000
description: >
  For my first blog post, I would like to write about Sharp-MAML, which combines SAM, a topic that has attracted considerable attention in generalization research since 2020, with MAML, one of the pioneering Meta-Learning algorithms. After briefly introducing SAM and MAML, I will explain what contribution comes from combining the two.
authors:
  - name: Jae-Jun Lee
    affiliations:
      name: UNIST
bibliography: blogs.bib
comments: true
hidden: false
tags: [Meta-Learning, MAML, Flat-Minima, SAM]
mathjax: true
_meta: >
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="[Paper Review] Sharp-MAML">
  <meta name="twitter:description" content="For my first blog post, I would like to write about Sharp-MAML, which combines SAM, a topic that has attracted considerable attention in generalization research since 2020, with MAML, one of the pioneering Meta-Learning algorithms. After briefly introducing SAM and MAML, I will explain what contribution comes from combining the two.">
---

## Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning

## 1. What is SAM?
SAM (Sharpness-Aware Minimization) was introduced in an ICLR 2021 paper and opened up a new perspective on generalization research at Google Research. The two main goals of SAM are:

* <strong>Improves Model Generalization via Finding Flat Minima</strong>
* <strong>Provides Robustness to Label Noise</strong>  

But what does it mean to find flat minima? Before moving on to the SAM algorithm, I will briefly explain what flat minima are.

### 1.1 Flat Minima  
Flat minima is one of the concepts that commonly comes up when discussing generalization in Deep Learning. From a Loss Landscape perspective, if the region around a minimum is flat, it generally indicates better generalization. On the other hand, if the region around a minimum is sharp, the model may perform well on a specific task but generalize less effectively. You can understand this more intuitively by looking at the figure below [1].

![img]({{ '/assets/img/23-03-13/flat_minima.png' | relative_url }}){: .post-image style="--post-image-width: 680px;"}

Figure [1]: <i>Example of Flat minimum & Sharp Minimum</i>

If you look at Figure 1 in Figure [1], the loss landscape is not deep but flat, while the loss landscape in Figure 2 is deep but narrow. The loss at the minimum point (<highlight style="color: red">red dot</highlight>) in Figure 1 will probably be higher than the loss at the minimum point in Figure 2. However, even if the model deviates slightly from the optimal point during training (<highlight style="color: blue">blue dot</highlight>), the difference in loss will not be large in Figure 1 because the landscape is gentle.

However, in Figure 2, the landscape is steep, so even a slight deviation from the minimum results in a large difference in loss. In other words, **Figure 1** has a **small loss difference** between the red dot and blue dot positions, while **Figure 2** has a **large loss difference** between them. Since there is no guarantee that Deep Learning will always find the optimal point during training, a flatter loss landscape can be considered more advantageous from a generalization perspective.

(However... there is another perspective on whether flatness really has a significant impact on generalization. See Li, Hao, et al., "Visualizing the loss landscape of neural nets." <i>Advances in neural information processing systems 31 (2018).</i> [Paper](https://arxiv.org/abs/1712.09913 "Paper"))

### 1.2 SAM Algorithm
So, how can we make the loss landscape flat?
The SAM algorithm finds flat minima through the following four steps. (I will omit the proof using formulas.)

1. Calculate Loss: $$\mathcal{L}_\mathcal{B}$$
2. Move in the <strong>"+"</strong> gradient direction of the calculated Loss:  $$\hat{\epsilon}(w)\rightarrow$$ $$w_{adv}=w_t+\hat{\epsilon}(w)$$  
(= Move to the highest Loss)
3. Calculate gradient **at the moved position**:
$$g =\nabla\mathcal{L}_\mathcal{B}(w)|_{w+\hat{\epsilon}(w)}$$
4. Proceed with weight update from the original position: $$w_t=w_t-\eta g$$  

>  *For steps 2 and 3, the paper provides a detailed proof. During the operations involving $$\epsilon$$, the derivation uses a Taylor expansion. Since this is not a SAM paper review... if you are interested, it would be a good idea to look at the paper directly. I will review it if I have a chance next time. ([https://arxiv.org/abs/2010.01412](https://arxiv.org/abs/2010.01412))

You can think of the core of SAM as an algorithm based on min-max optimization. Usually, when updating parameters during training, we use the form $$w = w - \nabla_w{L(w)}$$. We update in the opposite direction of the gradient calculated from the Loss. However, in the SAM algorithm, we first move in the positive direction of the gradient. The idea is that, even if we give up finding the lowest loss directly, we can update the gradient while considering the direction that lowers the highest losses. (Refer to Figure [2])

To explain it a bit more intuitively, SAM updates the model while pressing down on the highest losses. In other words, it makes the loss landscape generally flat by lowering the high losses rather than simply searching for the lowest loss.

![img]({{ '/assets/img/23-03-13/SAM_Algorithm.png' | relative_url }}){: .post-image}

Figure [2]: <i align='left'>SAM Algorithm</i>

Since SAM is an optimization algorithm rather than a specific model architecture, it can be applied to various models. (It can be used like an optimizer.)

## 2. What is MAML?
MAML (Model-Agnostic Meta-Learning) was introduced in a 2017 paper by Professor Chelsea Finn (who was a PhD student at the time), and it is one of the papers that helped establish Meta-Learning. Since it is such a famous paper, many people probably already know it, but I will briefly explain what Meta-Learning is and what kind of algorithm MAML is.

### 2.1 Meta-Learning
Before that, what is Meta-Learning? Meta-Learning is one of the fields related to few-shot learning. It is a learning method that focuses on learning across various tasks rather than learning only from Labels (supervised learning), so that the model can classify or predict well even when it encounters a completely new task. In other words, it learns how to adapt to new tasks quickly. (Task: sampling **K** items from each of **N types** of data in a dataset → **N-way K-shot**)

> Before understanding Meta-Learning, it is helpful to first understand few-shot learning. Since Few-Shot Learning is explained well in [this blog](https://zzaebok.github.io/machine_learning/FSL/), I will refer you to that post instead. Here, I think it is enough to briefly understand the concepts of the Support and Query sets.

Meta-Learning methods are usually divided into the following three types:
- <strong>model-based</strong> :arrow_right: Learning centered on the task's model
- <strong>metric-based (non-parametric)</strong> :arrow_right: Learning centered on distances between task representations
- <strong>gradient-based (parametric)</strong> :arrow_right: Learning centered on the gradients of the task parameters

<strong style="font-size: 0.9rem; font-style: normal">As a side note, the preferred method varies depending on the application.</strong>
- In **computer vision applications**, such as few-shot classification, **metric-based learning** is often used.
- In **RL applications**, such as robotics, **model-based learning** is often used.
- Since **gradient-based learning** directly learns the model parameters, it is widely used in **various fields**.

Meta-Learning also includes meta-training and meta-validation/test processes.

### 2.2 MAML
MAML belongs to gradient-based Meta-Learning among the methods classified above. MAML is referenced in many Meta-Learning papers because it is simple and convenient to use. As its name suggests, it can be used with any model (Model-Agnostic) and can adapt quickly (fast adaptation) to various tasks. The core processes that make up MAML are the following two:

* Calculate the loss for each task through fine-tuning from the initialized $\theta$ <strong>(Inner-Loop)</strong>
* Update the gradient of $\theta$ using the calculated loss <strong>(Outer-Loop)</strong>

> Usually, this process is called **bi-level optimization**. 

Ultimately, MAML's goal is to move the initialized $\theta$ to a position that can perform well on many different tasks. To do so, it goes through the two processes mentioned above, which I will explain in detail using Figure [3] and the notation below.

<div class="post-image-pair">
  <img src="{{ '/assets/img/23-03-13/MAML_Diagram.png' | relative_url }}" alt="MAML update diagram">
  <img src="{{ '/assets/img/23-03-13/MAML_algo.png' | relative_url }}" alt="MAML algorithm">
</div>

Figure [3]: <i>Diagram and Algorithm of MAML  </i>

<strong>Notation </strong>:  
- $\mathcal{T_i} = (\mathcal{S_i}, \mathcal{Q_i})$ : (Support, Query) datasets  
- $\theta$ : Initialized parameter  
- $\theta^\prime$ or $\phi$ : fine-tuned parameter  
- $\nabla \mathcal{L_i}$ : gradient of loss from fine-tuned parameter  
- *(Notation may differ from other blogs and papers.)* As mentioned above, to understand MAML, you need to understand two processes: the Inner and Outer Loops.

#### - Inner-Loop & Outer-Loop
First is the Inner-Loop. The Inner-Loop is the process of finding a good point for a given task through fine-tuning. Expressing this process as a formula, it is $\theta^{\prime} = \theta - \alpha \nabla_{\theta} \mathcal{L}(\theta) $. (This is the same as step 6 on the right of Figure [3].) This process is identical to *Stochastic* *Gradient* *Descent*. In other words, it quickly finds a good point for the task through SGD. In the MAML paper, this process is performed for 5 steps. Mathematically, from step 2 onward, the parameter in the SGD equation above changes from $\theta$ to $\theta^{\prime}$.
The detailed process of the Inner-Loop is as follows:
1. Sample $\mathcal{T_i}$: ($\mathcal{S_i}$, $\mathcal{Q_i}$) from distribution $\mathcal{p(T)}$
2. Iterate $\theta^{\prime} ← \theta - \alpha \nabla_{\theta} \mathcal{L}(\theta)$ with the $\mathcal{S_i}$ N times (= fine-tuning)
3. Calculate the $\mathcal{L}(\theta^{\prime})$ using the $\mathcal{Q_i}$

To explain the process above more simply: fine-tune with the Support set, and then check the performance at the fine-tuned position using the Query set—that is, calculate the loss.

Next is the Outer-Loop. The Outer-Loop is the process of updating $\theta$ using the average of the losses calculated in the Inner-Loop. The key point is that the update is performed not at the fine-tuned point, but **at the point where fine-tuning started: $\theta$**. Expressing this process as a formula, it is $\theta = \theta - \beta \nabla_{\theta} \sum \mathcal{L}(\theta^{\prime}) $. (This is the same as step 8 on the right of Figure [3].) The meaning of this formula can be understood as follows:
1. Use $\mathcal{S_i}$ to provide information about the corresponding task and train the model. (= Fine-tuning)
2. After fine-tuning, use $\mathcal{Q_i}$ to evaluate the performance on the corresponding task and calculate the Loss. (= Calculate Loss)
3. Determine the direction in which $\theta$ should learn using the Gradient of the Loss calculated in step 2.
4. Update based on the average of the loss gradients for $n$ tasks. ($i = 1,2,...,n$)

You can think of the gradient of the loss calculated with $\mathcal{Q_i}$ (the gradient vector) as indicating the direction of a future update. Since training on an unseen task (here, $\mathcal{Q_i}$) from the beginning is difficult and inefficient, we first adapt to some extent with $\mathcal{S_i}$ and then use the gradient obtained from the unseen task. Unlike supervised learning, which learns individual tasks or data one by one, MAML can be seen as **learning how to move toward an optimal point**. Since it does not learn only from specific data, overfitting is less likely, and it also generalizes well because it can quickly adapt to unseen tasks.

*However, there is also a disadvantage: the computational cost is relatively high because differentiation is performed twice (**Inner-Loop differentiation**, **Outer-Loop differentiation** $\rightarrow$ **Hessian**).*

> From now on:\\
> Inner-Loop = <strong>Fine-tuning </strong>  
> Outer-Loop = <strong>Meta-update </strong>  


## 3. Sharp-MAML?

You can think of the Sharp-MAML paper as a combination of SAM and MAML, as described above.

### 3.1 Problem Formulation & Algorithm

![img]({{ '/assets/img/23-03-13/Sharp-MAML_formulation.png' | relative_url }}){: .post-image style="--post-image-width: 640px;"}

> 1. Apply only during Fine-Tuning: $\alpha_{up} = 0$ & $\alpha_{low} > 0$  
> 2. Apply only during Meta-update: $\alpha_{up} > 0$ & $\alpha_{low} = 0$  
> 3. Apply Both:  $\alpha_{up} > 0$ & $\alpha_{low} > 0$  

Figure [4]: <i>Problem formulation of Sharp-MAML</i>

Before diving in!!  
- In this paper, the Taylor approximation used in SAM is interpreted as biased mini-batch gradient descent (BGD). (At the point: $\theta + \epsilon + \epsilon_m$)
- BGD$(\theta,\epsilon, \epsilon_m) = \theta + \epsilon - \beta_{low} \nabla \mathcal{L}(\theta + \epsilon + \epsilon_m) $ 

If you look at the right side of Figure [4], you can see how SAM was applied to MAML. It was applied in three ways: during fine-tuning, during the meta-update, or during both.
First, let's look at the lower part. During each one-step update in fine-tuning, the authors perturb the surroundings to find a high-loss point and then proceed in the direction that lowers that loss. The formula in the paper is as follows.



- For each task m...
- perturbation: $\epsilon_{m}(\theta) = \alpha_{low} \nabla \mathcal{L}(\theta) / \|\|{\nabla \mathcal{L}(\theta)}\|\|_{2}$
- maximum point (=maximum loss): $\theta + \underset{\|\|\epsilon_m\|\|\_{2} \leq \alpha_{low}}{\max} \epsilon_{m}(\theta)$  
- gradient descent: $\tilde{\theta^1} = BGD(\theta, 0, \epsilon_m) = \theta -\beta_{low}\nabla \mathcal{L}(\theta + \epsilon_{m}(\theta))$
- regularizer term: $\frac{\|\| \theta_m - \theta \|\|}{\beta_{low}}$   

Next is the upper part. Here too, we add a perturbation to find the highest loss within that range and then proceed in the direction that lowers that loss. The difference is that, when calculating the perturbation, we use the gradient calculated during fine-tuning. The formula in the paper is as follows.

- (meta) perturbation: $$\epsilon(\theta) = \alpha_{up} \nabla \mathcal{h} / \|\mathcal{h}\|_2$$(→$$\nabla \mathcal{h} = \nabla_{\theta} \sum_{m=1}^{M}\mathcal{L}(\tilde{\theta^1})$$)
- maximum point: $\theta + \underset{\|\|\epsilon_m\|\|\_{2} \leq \alpha_{low}}{\max} \epsilon_{m}(\theta) + \epsilon(\theta)$
- gradient descent: $\tilde{\theta^2} = BGD(\theta, \epsilon, \epsilon_m) = \theta + \epsilon - \beta_{low} \nabla \mathcal{L}(\theta + \epsilon + \epsilon_m)$
- meta-update: $\theta \leftarrow \theta - \beta_{up} \sum_{m=1}^M \nabla_{\theta}\mathcal{L}(\tilde{\theta^2}) $

The above process is shown in pseudo-code in Figure [5].

![image.png1]({{ '/assets/img/23-03-13/Sharp-MAML_algorithm.png' | relative_url }}){: .post-image style="--post-image-width: 640px;"}
Figure [5]: <i>Pseudo-Code for Sharp-MAML</i>

### 3.2 Results

The results are as follows.

<div class="post-image-pair">
  <img src="{{ '/assets/img/23-03-13/Sharp-MAML_result_1shots.png' | relative_url }}" alt="Sharp-MAML one-shot results">
  <img src="{{ '/assets/img/23-03-13/Sharp-MAML_result_5shots.png' | relative_url }}" alt="Sharp-MAML five-shot results">
</div>

Figure [6]: <i>Results of Sharp-MAML</i>

It was somewhat disappointing. There is a gain of about 2–3%, but compared to the Meta-Learning papers being published these days, it is not a huge gain. Actually, when I first read this paper (around May 2022...), the result in the officially published paper was around 60% for 5-way 1-shot, but when I looked it up again recently, it had changed to 50%. In terms of novelty, there is definitely a contribution, but since the results are not outstanding, I think it would have been a stronger paper if the gains had been larger. Also, if the authors wanted to emphasize generalization more, I wonder what the results would have looked like if they had included cross-domain adaptation as well.

Finally, let's briefly examine what Sharp-MAML means from the perspective of its loss landscape. (These are also my subjective thoughts.)
![img]({{ '/assets/img/23-03-13/Sharp-MAML_Diagram.png' | relative_url }}){: .post-image}

Figure [7]: <i>Loss Landscape of Sharp-MAML</i>

Looking at Figure [7], you can see that the loss landscape has become considerably flatter than that of the original MAML. The MAML paper showed that MAML has strengths in generalization, so I was curious: "Is MAML also flat?" However, its loss landscape was not particularly flat. This leads to the question: "Does generalization improve if MAML's loss landscape becomes flatter?" Sharp-MAML provides results related to this question. If we think about it carefully, when the loss landscape is flat, the possibility of different tasks falling into local minima during fine-tuning or the meta-update may decrease. Therefore, it is reasonable to expect generalization to improve.

## 4. Conclusion

Just as I was becoming interested in Flatness while studying Meta-Learning, I came across the Sharp-MAML paper. It was impressive to see the authors address MAML through flatness from a generalization perspective. However, since the method is gradient-based, it still seems insufficient to overcome the limitations of the black-box (?) nature of these models, no matter how interesting the novelty is. More research seems necessary. For those who want to understand the novelty of this paper in more detail, it would be a good idea to read the theoretical analysis or the appendix in the paper.

## 5. Reference

* Abbas, Momin, et al. "Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning." International Conference on Machine Learning. PMLR, 2022. ([Paper](https://proceedings.mlr.press/v162/abbas22b/abbas22b.pdf))
* Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." International conference on machine learning. PMLR, 2017. ([Paper](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf))
* Foret, Pierre, et al. "Sharpness-aware minimization for efficiently improving generalization." arXiv preprint arXiv:2010.01412 (2020). ([Paper](https://arxiv.org/pdf/2010.01412))
* Li, Hao, et al. "Visualizing the loss landscape of neural nets." Advances in neural information processing systems 31 (2018). ([Paper](https://arxiv.org/abs/1712.09913))
