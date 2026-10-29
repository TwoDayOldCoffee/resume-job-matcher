# Resume Job Matcher

Upload a resume and get the 10 best-fitting jobs from 17,145 real LinkedIn postings, each with the reasons it fits and the requirements you don't yet show.

**Live demo:** https://resume-job-matcher-60gv.onrender.com (free hosting: the first visit after a quiet spell takes about a minute to wake up; the sample buttons work without uploading anything)

## How it works

```
resume ──► LLM profile ──► semantic search ──► LLM ranking ──► grounded explanations
           title, seniority,   bge-small embeddings,  top 20 reordered,   why each top-5 job fits
           skills, domains     seniority penalty      best 10 kept        + gaps from the posting
```

1. **Read.** `gpt-oss-20b` turns the resume into a structured profile. A long resume is a poor search query on its own: 512-token embedding models cut it off, often before the skills section.
2. **Search.** The raw resume and the profile are both embedded with `bge-small-en-v1.5` and averaged; the 20 closest postings are retrieved, with jobs at the wrong experience level demoted.
3. **Rank.** The LLM reorders those 20 by real fit and keeps the best 10.
4. **Explain.** A separate call writes, for the top 5, why each job fits and which stated requirements the resume doesn't show.

## Results

Evaluated on 210 resumes across 14 job categories.

**The benchmark labels were measuring the wrong thing.** The resume dataset's categories often describe the industry a person worked in, not their role (a resume labelled "ADVOCATE" belongs to a registered nurse), so correct matches were scored as wrong. I dropped the 9 categories whose resume titles don't match their label (decided from the titles, before scoring), and added an **LLM judge** (`gpt-oss-120b`, never shown the labels) that grades each retrieved job 0 / 1 / 2. Checked by hand, the judge agreed with my reading in about 6 of the 8 sharpest disagreements with the labels; 69% of label "misses" were judged at least a partial match.

| Version | Label P@10 (210 resumes) | Judge P@10, good match (42 resumes) | Kept? |
|---|---|---|---|
| Random | 0.106 | | |
| Raw resume as query | 0.706 | | |
| Raw resume + LLM profile | 0.742 | 0.452 | yes |
| + chunked job postings | 0.742 | 0.414 | no |
| + cross-encoder reranker (bge-reranker-base) | 0.741 | 0.412 | no |
| + seniority penalty | 0.747 | 0.452 | yes |
| + boilerplate removal | 0.751 | 0.443 | no |
| **+ LLM ranking of the top 20** | **0.745** | **0.500** | **yes** |

- **LLM ranking is the only change that measurably improved fit:** better on 15 resumes, worse on 4 (sign test, p = 0.019). It pulls about 2.4 jobs per list up from ranks 11–20. In this run 3 of 42 ranking responses were malformed and fell back to search order; the score counts them as they are, and the app now retries a failed call.
- **Negative results:** chunking postings, an off-the-shelf cross-encoder (trained for search relevance, not job fit) and boilerplate removal did not help, so they are not in the app.
- **Explanations are grounded:** a dedicated explanation call cut "no gaps listed" from 82% to 4% of explanations, and 97% of the listed gaps appear in the posting text.
- The judge and the ranker are different models from the same family (gpt-oss), so some shared bias can't be ruled out.

## Deployment

The app runs on Render's free tier (512 MB RAM). Two changes made it fit:
- **ONNX Runtime instead of PyTorch** for the embedding model (`fastembed`). Same model and weights: embeddings match (cosine ≈ 1.000) and top-20 retrieval overlaps 19.9/20 on the evaluation set.
- **No pandas or pyarrow in the app:** jobs load from compressed JSON. Peak memory dropped from 529 MB to 408 MB (measured locally).

Uploaded resumes are processed in memory and never stored; they are sent to Groq for the LLM steps. LLM output is escaped before display, and the prompts treat resume text as data.

## Repository

| Path | Contents |
|---|---|
| [`01_preprocessing.ipynb`](01_preprocessing.ipynb) | Downloads the data, labels job postings with categories, builds the corpus and evaluation sets |
| [`02_retrieval.ipynb`](02_retrieval.ipynb) | Embeddings and every retrieval experiment, scored with label P@10 |
| [`03_llm.ipynb`](03_llm.ipynb) | Resume profiles, the LLM judge, LLM ranking, explanation checks, data export for the app |
| [`app/`](app) | The deployed app: FastAPI backend (`app.py`), pipeline (`pipeline.py`), web page, Dockerfile |

## Run it locally

```bash
cd app
pip install -r requirements.txt
echo GROQ_API_KEY=your-key > .env      # free key from console.groq.com
uvicorn app:app --port 8000
```

Then open http://localhost:8000. The notebooks need the two Kaggle datasets below; set `BASE_DIR` at the top of each.

## Tech

Python, FastAPI, sentence-transformers / ONNX Runtime (`fastembed`), `BAAI/bge-small-en-v1.5`, gpt-oss-20b and gpt-oss-120b via Groq, NumPy, pandas, Docker, Render.

## Data and licences

- Job postings: [LinkedIn Job Postings (2023-24)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) by arshkon, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The files in `app/data/` are derived from it (deduplicated, sampled, boilerplate removed, descriptions shortened) and are shared under the same licence. The postings are from 2023-24 and are not live listings.
- Resumes: [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) by snehaanbhawal, used for evaluation and the four demo samples.
