---
title: Resume Job Matcher
emoji: 📄
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
---

# Resume Job Matcher

Upload a resume and get the ten best-fitting jobs from 17,145 LinkedIn postings, each with reasons it fits and the gaps to close.

**How it works:** an LLM extracts a structured profile from the resume; semantic search (bge-small embeddings, adjusted for seniority) finds the 20 closest postings; an LLM reranks them; and grounded explanations are written for the top 5.

Built with FastAPI, sentence-transformers and gpt-oss via Groq. Job postings are from a public LinkedIn dataset (2023-24) and are not live listings.

## Data

Job postings: [LinkedIn Job Postings (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) by arshkon on Kaggle, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The files in `data/` are derived from it (deduplicated, sampled, boilerplate removed) and are shared under the same license.
