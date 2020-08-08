---
layout: default
permalink: /posts-by-tags/
title: "Posts by tags"
author_profile: true
header:
  image: "/images/data_art.png"
---

{% include group-by-array collection=site.posts field='tags' %}

<ul>
  {% for tag in group_names %}
    {% assign posts = group_items[forloop.index0] %}
    <li>
      <h2>{{ tag }}</h2>
      <ul>
        {% for post in posts %}
        <li>
          <a href='{{ site.baseurl }}{{ post.url }}'>{{ post.title }}</a>
        </li>
        {% endfor %}
      </ul>
    </li>
    
  {% endfor %}
</ul>
