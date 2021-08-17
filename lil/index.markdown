---
layout: page
title: "Microservice description language"
permalink: /lil/
---

Define metadata for:
- storages
- services
- pipelines and 
- business objects
and tie everything together.

Useful for documenting existing services, and designing new data architecture.

Helps answering questions
- "where does this data come from?"
- "do we have historical states of that object?"
- "should we make daily snapshot of a table, or stream updates"
- "what's the data retention policy for that table" 
- "how do we calculate this feature offline and online in a consistent manner"
, and other fun questions that arise when deploying model to production.

# Menu
{% include menu.markdown %}
