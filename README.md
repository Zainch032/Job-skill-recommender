# 🧠 AI Jobs Skill Recommender & Market Analysis
### 2025–2026 AI Job Market · Python · Pandas · Scikit-learn · Streamlit

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://job-skill-recommender-032.streamlit.app/)

> End-to-end data science project — from raw CSV to interactive web app — uncovering what skills, experience levels, and market forces actually drive AI salaries.

---

## ❓ Problem Statement

The AI job market is noisy. Candidates waste months learning the wrong skills, and hiring teams have no clear salary benchmark. This project answers:
- What skills do I actually need for a specific AI role?
- What *really* drives salary — experience, skills, or company size?
- Which AI roles are growing fast vs stagnating?

---

## 📦 Dataset
1,500+ real job postings · Fields: `job_title`, `required_skills`, `annual_salary_usd`, `experience_level`, `years_of_experience`, `company_size`, `industry`, `remote_work`, `demand_score`, `demand_growth_yoy_pct`, `ai_salary_premium_pct`

---

## 🔍 EDA & Analysis — 40%

### Salary Drivers
- Built a full **feature correlation bar chart** — `is_senior` title flag is the **#1 salary predictor**, beating raw years of experience
- `demand_score` and `demand_growth_yoy_pct` are strong price drivers — market scarcity beats tech stack labels
- `is_remote_friendly` and `benefits_score` show **near-zero** correlation — flexibility costs you nothing
- **Remote work has only 2.7% salary variance** across hybrid / remote / on-site — confirmed via grouped bar

| Experience Level | Avg Salary |
|---|---|
| Entry (0–2 yrs) | ~$130k |
| Mid (3–5 yrs) | ~$175k |
| Senior (6–9 yrs) | ~$220k |
| Lead (10+ yrs) | ~$250k |

### Skill Analysis (Demand vs Pay)
- Exploded pipe-separated skills, counted frequency, cross-referenced with avg salary per skill

| Tier | Skills | Avg Salary |
|---|---|---|
| 💰 Premium | Cloud, Leadership, Agile | $195k+ |
| ✅ Baseline | Python, SQL, Statistics | ~$185k–$197k |
| ⚠️ Commodity | Communication, Problem Solving | ~$128k–$165k |

- **Python** in **62.8%** of postings — #1 by demand by a massive margin
- **Cloud** pays the most (~$203k) despite not being #1 — scarcity creates price
- **SQL** is the most underpaid relative to demand — 2nd most required, near salary bottom

### Experience & Company Size
- Cross-tabulated `company_size × exp_bin` (normalized %) → grouped bar chart
- **6–9 years** is the #1 hired bracket across ALL company sizes (45–50% of postings)
- **Entry-level (0–2 yrs)** = less than 10% of the market everywhere — entry crunch is real
- **Startups** hire proportionally more 10+ year veterans than any other company size

### AI Engineering Deep Dive
- Pivot table: `job_title × experience_level` → avg salary → heatmap
- 2×2 bar grid: demand score, YoY growth, benefits, AI premium — all by AI job title
- **Lead ML Engineers** peak at **$310k**, **AI Agent Developers** at **$300k**
- **RAG Engineer** and **NLP Engineer** show the fastest YoY growth — emerging bottlenecks
- **AI Agent Developers**: top-3 demand + high growth, yet near-bottom for benefits

### Job Category × Industry
- Cross-tabulated `job_category × industry` (% by row) → `PuOr` heatmap
- **ML Operations in Government** leads at 17.6% concentration
- **Manufacturing** is the weakest sector for DS/ML professionals

---

## 🛠️ Feature Engineering — 25%

Raw skills were stored as a single pipe-separated string per row — unusable for modeling.

**5-Step Pipeline (`feature_engineer.py`)**

1. **Tokenize** — split `required_skills` on `|`, strip whitespace, explode into rows
2. **Select** — rank by frequency, keep top **93 skills** (excluded `Linux` — low signal)
3. **One-Hot Encode** — binary column per skill using vectorized `str.contains()`
4. **Merge Variants** — collapsed inconsistent naming via regex:

| Canonical | Variants Merged |
|---|---|
| `Cloud` | `Cloud (AWS/GCP/Azure)` |
| `Fine-Tuning` | `Fine-tuning`, `Fine_Tune`, `LLM Fine-tuning` |
| `LLM_APIs` | `LLM APIs`, `LLM Integration`, `GenAI APIs` |
| `Prompt_Engineering` | `Prompt Engineering`, `Prompt Design` |
| `Risk Analysis` | `Risk Management`, `Risk Assessment` |

5. **Save** → `cleaned_data.csv` — original 26 columns + 93 new binary skill columns

---

## 🤖 Recommender System — 20%

Content-based recommender using skill vectors (`model.py`):

1. **Job Profiles** — group by `job_title`, compute mean across all 93 skill columns → each title becomes a skill frequency vector
2. **Similarity Matrix** — cosine similarity across all job profiles → N×N matrix for O(1) lookup
3. **Recommend** — given a title: top N skills by frequency + top 3 similar roles by similarity score + avg salary & posting count
4. **Fuzzy Fallback** — partial string match if exact title not found, prevents hard failures

**Sample output:**
```
Data Scientist · $192,450 avg · 87 postings
Skills : Python 94% | SQL 78% | Statistics 65% | ML 61% | Cloud 54%
Similar: ML Engineer 91% | AI Researcher 84% | Data Engineer 79%
```

---

## ⚙️ Full Pipeline Flow

```
Raw CSV → feature_engineer.py → cleaned_data.csv
              ↓                        ↓
         analysis.py              model.py
         (16+ EDA charts)    (profiles + similarity)
              ↓                        ↓
                    app.py (Streamlit)
              ┌─────────────────────────────┐
              │  Tab 1: Skill Recommender   │
              │  Tab 2: Market Analysis     │
              │         (7 sections)        │
              └─────────────────────────────┘
```

---

## 🚀 Live Demo

**Try it now:** [https://job-skill-recommender-032.streamlit.app/](https://job-skill-recommender-032.streamlit.app/)

---

## 🏃 Run Locally

```bash
pip install -r requirements.txt && python feature_engineer.py && streamlit run app.py
```

*Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit*
