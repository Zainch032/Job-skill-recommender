import streamlit as st
from model import build_profiles, build_sim_matrix, recommend, load_dataset
from analysis import (
    load_data,
    plot_remote_work_salary,
    plot_experience_regression,
    plot_salary_by_experience_level,
    plot_experience_bracket_by_company_size,
    plot_salary_correlation_bar,
    plot_top_demand_vs_salary_skills,
    plot_high_paying_skills,
    plot_job_category_counts,
    plot_job_category_vs_industry,
    plot_ai_eng_salary_heatmap,
    plot_ai_eng_demand_metrics,
    plot_category_avg_salary,

    plot_category_skill_demand,
    plot_category_salary_by_education,
    plot_category_experience_transition,
    plot_category_summary_stats,
    get_all_categories,
)# In your app.py, import the new functions
  

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Jobs Skill Recommender",
    page_icon="🧠",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* Target fonts cleanly without breaking structural wrapper divs */
    html, body, p, div, span, label { 
        font-family: 'Inter', sans-serif; 
    }

    .hero {
        padding: 2.5rem 0 1.5rem 0;
        border-bottom: 1px solid #2e2e2e;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 1.9rem; font-weight: 600; color: #f0f0f0; letter-spacing: -0.02em; }
    .hero p  { font-size: 0.95rem; color: #888; margin: 0; }

    .stat-row { display: flex; gap: 1.5rem; margin: 1.5rem 0; }
    .stat-box {
        flex: 1;
        background: #1e1e1e;
        border: 1px solid #2e2e2e;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    .stat-label {
        font-size: 0.75rem; color: #666;
        text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500;
    }
    .stat-value {
        font-size: 1.4rem; font-weight: 600; color: #f0f0f0;
        margin-top: 0.2rem; font-family: 'DM Mono', monospace;
    }

    .section-label {
        font-size: 0.72rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em;
        color: #555; margin: 1.8rem 0 0.8rem 0;
    }

    .skill-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.55rem; }
    .skill-name { font-size: 0.88rem; color: #ddd; width: 160px; flex-shrink: 0; font-weight: 500; }
    .skill-bar-bg { flex: 1; height: 7px; background: #2a2a2a; border-radius: 99px; overflow: hidden; }
    .skill-bar-fill { height: 100%; background: #6366f1; border-radius: 99px; }
    .skill-pct { font-size: 0.78rem; color: #777; font-family: 'DM Mono', monospace; width: 35px; text-align: right; }

    .similar-card {
        display: flex; justify-content: space-between; align-items: center;
        background: #1e1e1e; border: 1px solid #2e2e2e;
        border-radius: 8px; padding: 0.8rem 1.1rem; margin-bottom: 0.5rem;
    }
    .similar-title { font-size: 0.88rem; font-weight: 500; color: #f0f0f0; }
    .similar-meta   { font-size: 0.78rem; color: #666; font-family: 'DM Mono', monospace; }
    .sim-badge {
        font-size: 0.72rem; background: #2a2a2a; color: #6366f1;
        padding: 0.2rem 0.5rem; border-radius: 99px; font-family: 'DM Mono', monospace;
        border: 1px solid #6366f1;
    }

    .chart-label {
        font-size: 0.72rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em;
        color: #555; margin: 2rem 0 0.5rem 0;
    }

    /* ── Dropdown Native Layout Restoration ── */
    div[data-baseweb="popover"] ul {
        max-height: 260px !important;
        overflow-y: auto !important;
    }

    div[data-baseweb="popover"] li {
        padding: 8px 14px !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def get_data():
    df           = load_dataset()
    job_profiles = build_profiles(df)
    sim_df       = build_sim_matrix(job_profiles)
    return df, job_profiles, sim_df

df, job_profiles, sim_df = get_data()


# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 AI Jobs Skill Recommender</h1>
    <p>Find the skills that matter most for any AI role — and explore the 2025/2026 market.</p>
</div>
""", unsafe_allow_html=True)

x = len(df)
st.markdown(f"Total number of jobs posted: **{x:,}**")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Recommender", "📊 Market Analysis"])


# ─────────────────────────────────────────────────────────────
# TAB 1 — RECOMMENDER
# ─────────────────────────────────────────────────────────────
with tab1:
    job_titles = sorted(df["job_title"].dropna().unique().tolist())
   
    col_sl, _ = st.columns([1, 3])
    with col_sl:
        top_n = st.slider("Number of skills to show", min_value=3, max_value=15, value=6)

    selected = st.selectbox(
        "Select a job title",
        options=job_titles,
        index=None,
        placeholder="Type to search or scroll...",
        key="job_selectbox"
    )

    st.markdown("---")

    if selected:
        result = recommend(selected, df, job_profiles, sim_df, top_n_skills=top_n)

        if result is None:
            st.warning(f"No data found for '{selected}'. Try another title.")
        else:
            # Stats
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-label">Avg Salary</div>
                    <div class="stat-value">${result['avg_salary']:,.0f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Job Postings</div>
                    <div class="stat-value">{result['postings']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top Skills
            st.markdown('<div class="section-label">Top Skills Required</div>', unsafe_allow_html=True)
            for skill, freq in result["top_skills"].items():
                fill = int(freq * 100)
                st.markdown(f"""
                <div class="skill-row">
                    <div class="skill-name">{skill}</div>
                    <div class="skill-bar-bg">
                        <div class="skill-bar-fill" style="width:{fill}%"></div>
                    </div>
                    <div class="skill-pct">{fill}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Similar Jobs
            st.markdown('<div class="section-label">Similar Job Titles</div>', unsafe_allow_html=True)
            for title, score in result["similar_jobs"].items():
                avg = df[df["job_title"] == title]["annual_salary_usd"].mean()
                st.markdown(f"""
                <div class="similar-card">
                    <div>
                        <div class="similar-title">{title}</div>
                        <div class="similar-meta">${avg:,.0f} avg salary</div>
                    </div>
                    <div class="sim-badge">{score:.0%} match</div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB 2 — MARKET ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab2:
    SECTIONS = [
        "📌 Market Overview",
         "📈  Deep Dive by category",
        "💰 Salary Drivers",
        "🛠️ Skill Analysis",
       
        "🏢 Experience & Company Size",
        "🤖 AI Engineering Deep Dive",
        "🌐 Job Category × Industry",
        
    ]

    df_analysis = load_data("data/ai_jobs_market_2025_2026.csv")

    section = st.radio("", SECTIONS, horizontal=True)

    st.markdown("---")

    # ── Market Overview ───────────────────────────────────────
    if section == "📌 Market Overview":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-label">Job Category Distribution</div>', unsafe_allow_html=True)
            st.pyplot(plot_job_category_counts(df_analysis))
        with col2:
            st.markdown('<div class="chart-label">Avg Salary by Job Category</div>', unsafe_allow_html=True)
            st.pyplot(plot_category_avg_salary(df_analysis))

    # ── Salary Drivers ────────────────────────────────────────
    elif section == "💰 Salary Drivers":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-label">Salary by Experience Level</div>', unsafe_allow_html=True)
            st.pyplot(plot_salary_by_experience_level(df_analysis))
        with col2:
            st.markdown('<div class="chart-label">Remote Work vs Salary</div>', unsafe_allow_html=True)
            st.pyplot(plot_remote_work_salary(df_analysis))
        st.markdown('<div class="chart-label">Years of Experience vs Salary (Regression)</div>', unsafe_allow_html=True)
        st.pyplot(plot_experience_regression(df_analysis))
        st.markdown('<div class="chart-label">Feature Correlation with Annual Salary</div>', unsafe_allow_html=True)
        st.pyplot(plot_salary_correlation_bar(df_analysis))
        st.markdown("""**Workforce Composition**
The data reflects a modern, flexible workforce where 75.4% of employees work in a non-traditional office setting.
- Hybrid: 45.7% (Majority)
- Fully Remote: 29.7%
- On-site: 24.6%

**Salary Analysis**
The financial data confirms that work location has a negligible impact on compensation. The variance between the highest and lowest average is only 2.7%""")
        st.markdown("""### Top 4 Strategic Insights

- **Seniority over Tenure:** The `is_senior` title is the strongest predictor of high pay, proving that organizational level impacts salary more than the raw count of `years_of_experience`.
- **Market Dynamics:** High `demand_score` and `demand_growth_yoy` are significant price drivers, suggesting that specialized market scarcity pushes salaries higher than specific tech labels like `is_llm_role`.
- **The Premium Paradox:** The slight negative correlation for `ai_salary_premium_pct` indicates that while entry-level roles get the biggest *percentage* raises, the highest *total* salaries remain in established senior brackets.
- **Seasonal & Cultural Neutrality:** Factors like `is_remote_friendly` and `benefits_score` show almost zero correlation with salary, meaning you can prioritize flexibility without sacrificing your earning potential.""")

    # ── Skill Analysis ────────────────────────────────────────
    elif section == "🛠️ Skill Analysis":
        st.markdown('<div class="chart-label">Top 10 In-Demand Skills vs Avg Salary</div>', unsafe_allow_html=True)
        st.pyplot(plot_top_demand_vs_salary_skills(df_analysis))
        st.markdown('<div class="chart-label">Top 10 Highest Paying Skills</div>', unsafe_allow_html=True)
        st.pyplot(plot_high_paying_skills(df_analysis))
        st.markdown("""### Key Takeaways

**From Chart 1 — Top 10 Most In-Demand Skills**

- **Python is the king** — 942 jobs (62.8%) making it appear in nearly 2 out of 3 job postings
- **SQL + Cloud together** cover ~59% of jobs — the core technical stack most employers expect
- **Soft skills are surprisingly common** — Communication, Leadership, Research all appear in ~25% of jobs
- **The gap is massive** — Python (942) vs Problem Solving (314) is a **3x difference**

**From Chart 2 — Avg Salary by Skill**

- **Cloud pays the most** (~$203k) despite not being #1 in demand — **scarcity drives price**
- **Communication pays the least** (~$128k) — the only skill below $180k
- **Python pays ~$197k** — high demand AND high salary makes it the single most valuable skill to learn
- **SQL is underpaid** — 2nd most required skill but near bottom in salary
- **Leadership + Cloud + Agile** are the premium tier — all above $195k

**The Combined Story**
Skills split into **3 tiers:**
- 💰 **Premium** → Cloud, Leadership, Agile ($195k+)
- ✅ **Baseline** → Python, SQL, Statistics (required but competitive)
- ⚠️ **Commodity** → Communication, Problem Solving (expected but don't boost salary)""")

    # ── Experience & Company ──────────────────────────────────
    elif section == "🏢 Experience & Company Size":
        st.markdown('<div class="chart-label">Experience Bracket by Company Size</div>', unsafe_allow_html=True)
        st.pyplot(plot_experience_bracket_by_company_size(df_analysis))
        st.markdown("""### Key Takeaways

- **The 6–9 Years Dominance:** This bracket is the undisputed priority for all company sizes, commanding 45% to 50% of all job openings.
- **The Entry-Level Crunch:** Opportunities for true juniors (0–2 years) are heavily restricted, making up less than 10% of the market everywhere.
- **The Bulk of Hiring:** Mid-to-senior professionals (3–9 years) form the core of the market, accounting for roughly 75% to 80% of total demand.
- **Startup Veteran Bias:** Startups (1–50 employees) actively hunt for deep expertise, leading all categories with the highest share of 10+ year roles.
- **Mid-Size Senior Peak:** Mid-size companies (501–5000 employees) are the most aggressive hirers of the 6–9 year bracket, peaking at nearly 50%.""")

    # ── AI Engineering ────────────────────────────────────────
    elif section == "🤖 AI Engineering Deep Dive":
        st.markdown('<div class="chart-label">Salary by Job Title & Experience Level</div>', unsafe_allow_html=True)
        st.pyplot(plot_ai_eng_salary_heatmap(df_analysis))
        st.markdown("""### Key Takeaways

- **Clear Seniority Premium:** Compensation scales predictably upward across all AI domains from entry-level to lead roles.
- **Peak Market Demand:** Lead-level **Senior ML Engineers** and **AI Agent Developers** top the market at $310k and $300k.
- **Entry-Level Inversion:** Senior ML Engineers entry-level baseline begins exceptionally high ($220k) before standard mid-level normalization.
- **Lower Bound Disciplines:** **AI Engineers** and **Prompt Engineers** consistently anchor the bottom of the pay structure (~$110k–$120k entry).
- **The $250k Senior Catalyst:** Reaching Senior status (6–9 years) acts as a standard threshold to break into the $240k–$250k tier.""")
        
        st.markdown('<div class="chart-label">Demand, Growth, Benefits & AI Premium</div>', unsafe_allow_html=True)
        st.pyplot(plot_ai_eng_demand_metrics(df_analysis))
        st.markdown("""### Key Takeaways

- **High Demand Stability:** Advanced backend AI specializations (**LLM Engineer** and **ML Engineer**) maintain the highest absolute market demand scores.
- **Aggressive Growth Trajectory:** Emerging niches like **RAG Engineer** and **NLP Engineer** are experiencing the most rapid year-over-year growth.
- **Perks Over Pay:** Organizations are compensating lower overall demand roles like **Generative AI Engineer** and **AI Engineer** with top-tier benefits packages.
- **The Emerging Bottleneck:** **AI Agent Developers** present a rare balanced constraint—ranking 3rd in total demand and exhibiting high growth, yet resting near the bottom for benefits.""")

    # ── Industry ──────────────────────────────────────────────
    elif section == "🌐 Job Category × Industry":
        st.markdown('<div class="chart-label">Role Distribution Across Industries (%)</div>', unsafe_allow_html=True)
        st.pyplot(plot_job_category_vs_industry(df_analysis))
        st.markdown("""### Key Takeaways

- **ML Operations in Government** commands the highest concentration (17.6), making it the most lucrative combination.
- **Architecture in Finance** (15.4) and **Infrastructure in Consulting** (16.4) are strong outliers — niche but high-paying intersections.
- **Research in Retail** (16.0) surprisingly outperforms many tech-adjacent roles, suggesting underrated demand.
- **AI Engineering** salaries are relatively consistent (6.2–10.2) across all industries — broad demand but no single dominant sector.
- **Data Science** pays best in **Automotive & Government** (11.0) and weakest in **Research** (3.1).
- **Manufacturing** shows low compensation for ML Operations (0.0) and Product roles (2.9) — least attractive sector for DS/ML professionals.
- **Security roles** are strong in Automotive, Education, and Finance (14.0) but nearly absent in Consulting (0.0).""")

            
# Then in your Data Science Deep Dive section (or any section):
    elif section == "📈  Deep Dive by category":
     categories = get_all_categories(df_analysis)
    
     selected_category = st.selectbox(
        "Select Job Category",
        options=categories,
        index=0,
        key="category_select"
    )
    
    # Get summary stats
     stats = plot_category_summary_stats(df_analysis, selected_category)
    
     if stats["count"] == 0:
        st.warning(f"No data found for '{selected_category}'")
     else:
        # Show stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Job Postings", f"{stats['count']:,}")
        with col2:
            st.metric("Avg Salary", f"${stats['avg_salary']:,.0f}")
        with col3:
            st.metric("AI Premium", f"{stats['ai_premium']}%")
        with col4:
            st.metric("Demand Score", f"{stats['demand_score']:.1f}")
        
     # Charts - Full width, stacked vertically
     st.markdown("---")

# Charts - First 2 in one row (40/60 split), last chart on second row
     st.markdown("---")

# Row 1: Two charts with custom width
     col1, col2 = st.columns([1, 1.3])  # 40% / 60% split

     with col1:
      st.markdown("#### 🔧 Top Skills Required")
      st.pyplot(plot_category_skill_demand(df_analysis, selected_category))

     with col2:
       st.markdown("#### 💰 Salary by Education Level")
       st.pyplot(plot_category_salary_by_education(df_analysis, selected_category))
    
    
     st.markdown("---")

# Row 2: Full width chart
     st.markdown("#### 📊 Experience Distribution by Job Title")
     st.pyplot(plot_category_experience_transition(df_analysis, selected_category))