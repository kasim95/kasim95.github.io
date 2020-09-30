---
layout: archive
permalink: /projects/
title: "Projects"
author_profile: true
read_time: false
header:
  image: "/assets/images/bgs/1.png"
  caption: "Photo Credit: [**Text Logo Design**](http://simpledesktops.com/browse/desktops/2015/mar/27/innovation/)"
classes: wide
---

*"Anyone who has never made a mistake has never tried anything new."* - Albert Einstein

{% for post in site.posts %}

---
  {% include archive-single.html %}
{% endfor %}
