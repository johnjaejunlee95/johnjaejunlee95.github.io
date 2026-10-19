---
layout: distill
title: "[Conceptual Background] Meta Learning (1) - Few-Shot Learning"
date: 2023-12-24 00:00:00 +0000
authors:
  - name: Jae-Jun Lee
    affiliations:
      name: UNIST
bibliography: blogs.bib
comments: true
hidden: false
tags: [Meta-Learning, Few-Shot Learning]
mathjax: true
_meta: >
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="[Conceptual Background] Meta Learning (1) - Few-Shot Learning">
  <meta name="twitter:description" content="I would like to introduce the basic concepts of Meta-Learning, a topic I have been putting off for a while. Over the past few years, Meta-Learning has been a hot topic and a main keyword at top-tier conferences. Considering the speed of the AI ecosystem, it has also been studied for quite a long time. However, perhaps because it has been studied for so long, this field seems to have become saturated recently. It has become difficult to grow the &quot;pie&quot; any further through research on Meta-Learning itself. Nevertheless, I believe the concept of Meta-Learning and its effectiveness are still important. In particular, foundation models such as LLMs and LMMs need to cover a wide variety of tasks, and much of their learning proceeds in the form of Few-Shot Learning. In a broader sense, they can be viewed as being trained through Meta-Learning. Therefore, I think it is important to understand Meta-Learning, or, more specifically, Few-Shot Learning. You do not need to know every related algorithm, but understanding how to define problems and set up training when using Meta-Learning can lead to more diverse approaches, so I decided to cover this topic.">
---

<div>I would like to introduce the basic concepts of Meta-Learning, a topic I have been putting off for a while. Over the past few years, Meta-Learning has been a hot topic and a main keyword at top-tier conferences. Considering the speed of the AI ecosystem, it has also been studied for quite a long time. However, perhaps because it has been studied for so long, this field seems to have become saturated recently. It has become difficult to grow the "pie" any further through research on Meta-Learning itself.
<br><br> Nevertheless, I believe the concept of Meta-Learning and its effectiveness are still important. In particular, foundation models such as <highlight style="color: red">LLM/LMM</highlight> need to cover a wide variety of tasks, and much of their learning proceeds in the form of Few-Shot Learning. In a broader sense, they can be viewed as being trained through Meta-Learning.
  <br><br> Therefore, I think it is important to understand Meta-Learning, or, more specifically, Few-Shot Learning. You do not need to know every related algorithm, but understanding how to define problems and set up training when using Meta-Learning can lead to more diverse approaches, so I decided to cover this topic.</div>



### 0. What is Meta Learning?
When explaining Meta-Learning, many people describe it as "Learn to Learn." Put simply, you can think of it as the way humans learn. As people live, they accumulate various experiences and acquire new knowledge based on them. We acquire new knowledge by combining what we already know with new experiences. Even if we are not taught everything from A to Z, we can make inferences based on the knowledge we already possess (of course, completely new things are exceptions). For example, even if someone cannot distinguish between an apple and a dog at first, a few experiences help them understand the characteristics of apples and dogs. Based on this, they become able to distinguish new apples or dogs. Later, even if they see an animal similar to a dog or a fruit similar to an apple, they can distinguish it. In this way, humans learn through 'experience', and Meta-Learning applies this ability to deep learning methods.

### 1. Few-Shot Learning
Then, let's try applying this process to AI model training.

First, let's think about conventional deep learning. Existing deep learning usually requires training on numerous images for each class, as shown in <a href='#figure1'>Figure 1</a> below. However, this approach can cause problems such as overfitting, and from a generalization perspective, it is difficult to find a solution other than providing more diverse data. Since data is always limited, gathering more and more diverse data is difficult and inefficient. So, what should we do to train deep learning models more efficiently, like humans? One of the concepts introduced to address this problem is "[Few-Shot Learning](https://en.wikipedia.org/wiki/Few-shot_learning)".

<center>
  <img class="post-image" style="--post-image-width: 680px;" src="{{ '/assets/img/23-12-24/supervised_learning.png' | relative_url }}"> <br>
  <figcaption>
    <a id='figure1'>Figure 1. Fundamental Process of Supervised Learning</a>
  </figcaption>
  <br>
</center>



To put Few-Shot Learning simply, it means learning from only a few data points at a time. In other words, it is a training method that uses only a few data points in each episode or epoch. Do not misunderstand this: it does not mean training with only a few data points in total. Rather, it means extracting a few data points from the entire dataset for each training step. Therefore, the setup is also slightly different from conventional deep learning methods. To understand the Few-Shot setting, you need to know the concepts of $N$-ways $K$-shots, as well as the Support Set ($S$) and Query Set ($Q$).

#### 1.1 $N$-Ways $K$-Shots

Here, $N$ means the number of classes to extract from the entire dataset (<a href='#figure2'>Figure 2</a>), while $K$ means the number of images to extract per class. For example, let's assume that a certain dataset has 100 classes. If the experimental setting is 5-way 1-shot, we randomly select 5 classes out of the 100 in each epoch and extract 1 image for each class.

<center>
  <img class="post-image" src="{{ '/assets/img/23-12-24/n_ways_k_shots.png' | relative_url }}">
  <br>
  <figcaption>
    <a id='figure2'>Figure 2. Example of N-ways K-shots</a>
  </figcaption>
</center>

#### 1.2 Support Set ($S$) and Query Set ($Q$)

First, before going further, you should think of $S$ and $Q$ as a bundle. In the AI community, this bundle is called an <b>episode</b> or a <b>task</b>. For this reason, few-shot learning is sometimes called <b>episodic learning</b>. During training, we sample episodes or tasks according to the batch size in each epoch.

Here, $S$ consists of the data that directly participate in training, while $Q$ consists of the data used to evaluate how well the model was trained. Those encountering few-shot learning for the first time might find this setting a bit confusing, so I will explain it using <a href='#figure3'>Figure 3</a> below.

<center>
  <img class="post-image" style="--post-image-width: 680px;" src="{{ '/assets/img/23-12-24/support_query.png' | relative_url }}">
  <figcaption>
    <a id='figure3'>Figure 3. An example of utilizing a support set and a query set.</a>
  </figcaption>
  <br>
</center>


If you look at <a href='#figure3'>Figure 3</a>, examples of $S$ and $Q$ are shown on the left. $S$ participates in training with its class labels known. In other words, we train the model with $S$ as in supervised learning. Therefore, $S$ follows the $N$-way $K$-shot setting. $Q$, on the other hand, is given to the trained model without using its class labels, and the result is used for evaluation.

One important point here is that while $Q$ follows the $N$-way setting, it does not necessarily follow the $K$-shot setting. The number of classes is fixed by $S$, but the number of images per class is not. In other words, we do not necessarily have to select $K$ images per class. Usually, the Query set contains 15 data points per class.

Additionally, we usually refer to $S$ as the seen task and $Q$ as the unseen task. Ultimately, the goal of few-shot learning is to achieve good performance on unseen tasks, so the model is evaluated and updated based on $Q$.

#### 1.3 Wrap-up

To summarize Few-Shot Learning, the process is as follows:

1. **Sample a task(= [$S$, $Q$]) according to $N$-ways $K$-shots.**
2. **Train with $S$ according to the method.**
3. **Evaluate whether it was trained well with $Q$.**
4. **Extract the final loss from the result in 3.**
5. **Update the model with the loss from 4.**
6. **Repeat 1~5 for the number of epochs.**

When training is done this way, the class labels change from task to task, which encourages the model to learn *how to learn* rather than memorizing each specific class. For example, in <a href='#figure2'>Figure 2</a>, 'car' may be labeled as class 1, but in another epoch it could be class 2 or class 3. In other words, the model develops the ability to recognize cars rather than memorizing a fixed label for 'car'. As a result, even when a completely new task appears, the model can distinguish it to some extent. This is beneficial from a Generalization perspective.

As a side note, these days, most LLMs and LMMs are trained with few-shot or zero-shot learning. (I will post about zero-shot learning separately when I have a chance.) In fact, the strength of LLMs and LMMs is not simply performing well in one specific situation, but performing well across various situations. In this case, it is reasonable to train them with few-shot or zero-shot learning—that is, to teach them how to distinguish between different situations. If I have the chance, I will explain this in more detail in a future post about LLMs and LMMs.



### *Future Series

Trying to make this post as detailed as possible made it longer than expected. Originally, I intended to include everything in one post, but there were more parts to explain and more content to cover than I thought. So, I think it is better not to make it any longer. I will wrap up here for now and continue posting about Meta-Learning approaches, methods, and significance in a series later. Thank you for reading.
