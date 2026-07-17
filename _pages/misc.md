---
layout: page
permalink: /misc/
title: Misc.
description: Education, experience, projects, patents, teaching, and skills.
nav: true
nav_order: 3
---

## education

<div class="about-list">
  {% for item in site.data.education %}
    <div class="about-list-item">
      <div>
        <strong>{{ item.school }}</strong>
        <span>{{ item.degree }}, {{ item.department }}</span>
      </div>
      <div class="about-meta">{{ item.start }} - {{ item.end }}{% if item.gpa %} | GPA {{ item.gpa }}{% endif %}</div>
    </div>
  {% endfor %}
</div>

## experience

<div class="about-list">
  {% for item in site.data.experiences %}
    <div class="about-list-item">
      <div>
        <strong>{{ item.role }}</strong>
        <span>
          {% if item.lab_url %}<a href="{{ item.lab_url }}">{{ item.lab }}</a>{% else %}{{ item.lab }}{% endif %}, {{ item.org }}
        </span>
      </div>
      <div class="about-meta">{{ item.start }} - {{ item.end }}</div>
    </div>
  {% endfor %}
</div>

## projects

<div class="about-list">
  {% for item in site.data.projects %}
    <div class="about-list-item">
      <div>
        <strong>{{ item.title }}</strong>
        <span>{{ item.sponsor }}</span>
      </div>
      <div class="about-meta">{{ item.start }} - {{ item.end }} | {{ item.role }}</div>
    </div>
  {% endfor %}
</div>

## patents

<div class="about-list">
  {% for item in site.data.patents %}
    <div class="about-list-item">
      <div>
        <strong>{{ item.title }}</strong>
        <span>{{ item.number }}</span>
      </div>
      <div class="about-meta">{{ item.publication_date }}</div>
    </div>
  {% endfor %}
</div>

## teaching

<div class="about-list">
  {% for item in site.data.teaching %}
    <div class="about-list-item">
      <div>
        <strong>{{ item.title }}</strong>
        <span>{{ item.role }}, {{ item.org }}</span>
      </div>
      <div class="about-meta">{{ item.start }} - {{ item.end }}</div>
    </div>
  {% endfor %}
</div>

## skills

<div class="about-list">
  {% for group in site.data.skills %}
    <div class="about-list-item">
      <div>
        <strong>{{ group.category }}</strong>
        <span>{{ group.items | join: ", " }}</span>
      </div>
    </div>
  {% endfor %}
</div>
