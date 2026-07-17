---
layout: about
title: about
permalink: /
subtitle: Ph.D. Candidate @ Graduate School of Artificial Intelligence, UNIST
profile:
  image: profile.jpg
  image_cicular: false # crops the image to make it circular
  address: false

news: false # includes a list of news items
selected_papers: true # includes a list of papers marked as "selected={true}"
social: true  # includes social icons at the bottom of the page
---

I am a fifth-year Ph.D. candidate whose research focuses on understanding how AI models learn and generalize from a theoretical perspective. My previous work centered on meta-learning, while my current research interests lie in multimodal learning, robustness, and generalization. More recently, I have also been exploring AI applications for biomedical data.

<div class="about-keywords">
  {% for keyword in site.data.home_profile.focus_keywords %}
    <span>{{ keyword }}</span>
  {% endfor %}
</div>
