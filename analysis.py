"""
analysis.py
-----------
All EDA, chart, and analysis functions for the AI Jobs Salary project.
Each function returns a Matplotlib Figure — call st.pyplot(fig) in app.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load the AI jobs CSV and return a cleaned DataFrame."""
    df = pd.read_csv(filepath)
    df.drop(columns=["job_id"], inplace=True, errors="ignore")

    # Skill list column
    df["skills_list"] = df["required_skills"].str.split("|")

    # Experience bins
    df["exp_bin"] = pd.cut(
        df["years_of_experience"],
        bins=[0, 2, 5, 9, 15],
        labels=["0-2", "3-5", "6-9", "10+"],
    )
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset stats as a dict for display in Streamlit."""
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "duplicates": df.duplicated().sum(),
        "nulls": df.isnull().sum().sum(),
        "salary_mean": round(df["annual_salary_usd"].mean(), 2),
        "salary_min": df["annual_salary_usd"].min(),
        "salary_max": df["annual_salary_usd"].max(),
    }


# ---------------------------------------------------------------------------
# 1. Remote Work
# ---------------------------------------------------------------------------

def plot_remote_work_salary(df: pd.DataFrame) -> plt.Figure:
    """Bar chart: avg salary by remote work type."""
    data = df.groupby("remote_work")["annual_salary_usd"].mean().round(2).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=data, x="remote_work", y="annual_salary_usd", palette="Set2", ax=ax)
    ax.set_title("Average Salary by Remote Work Type")
    ax.set_xlabel("Remote Work Policy")
    ax.set_ylabel("Avg Annual Salary (USD)")
    ax.set_ylim(170000,200000)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Experience vs Salary
# ---------------------------------------------------------------------------

def plot_experience_regression(df: pd.DataFrame) -> plt.Figure:
    """Regression plot: years of experience vs annual salary."""
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.regplot(data=df, x="years_of_experience", y="annual_salary_usd", ax=ax,
                scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
    ax.set_title("Years of Experience vs Annual Salary")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Annual Salary (USD)")
    plt.tight_layout()
    return fig


def plot_salary_by_experience_level(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar: avg salary by job-assigned experience level."""
    order = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"]
    data = (
        df.groupby("experience_level")["annual_salary_usd"]
        .mean()
        .reindex(order)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=data, x="annual_salary_usd", y="experience_level",
                palette="Purples_r", ax=ax)
    ax.set_title("Avg Salary by Experience Level")
    ax.set_xlabel("Annual Salary (USD)")
    ax.set_ylabel("Experience Level")
    ax.set_xlim(120_000, 260_000)
    plt.tight_layout()
    return fig


def plot_experience_bracket_by_company_size(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar: experience bracket distribution by company size."""
    ct = round(
        pd.crosstab(df["company_size"], df["exp_bin"], normalize="index") * 100, 2
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ct.plot(
        kind="bar",
        color=["#1f4e79", "#2e8b8b", "#5b9279", "#8faabf"],
        ax=ax,
    )
    ax.set_title("Experience Bracket Distribution by Company Size")
    ax.set_ylabel("% of Jobs")
    ax.set_xlabel("Company Size")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontweight="bold")
    ax.legend(title="Years of Exp", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Salary Correlation
# ---------------------------------------------------------------------------

def plot_salary_correlation_bar(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar: correlation of all features with annual salary."""
    drop_cols = ["posting_year", "posting_month", "salary_min_usd", "salary_max_usd"]
    corr = (
        df.select_dtypes(include="number")
        .drop(columns=drop_cols, errors="ignore")
        .corr()["annual_salary_usd"]
        .sort_values(ascending=False)
        .drop("annual_salary_usd")
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=corr.values, y=corr.index, palette="RdBu", ax=ax)
    ax.set_title("Feature Correlation with Annual Salary")
    ax.set_xlabel("Correlation Coefficient")
    plt.tight_layout()
    return fig


def plot_salary_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap: single-column correlation with annual salary (detailed view)."""
    drop_cols = ["posting_year", "posting_month", "salary_min_usd", "salary_max_usd"]
    corr_series = (
        df.select_dtypes(include="number")
        .drop(columns=drop_cols, errors="ignore")
        .corr()["annual_salary_usd"]
        .sort_values(ascending=False)
        .drop("annual_salary_usd")
    )

    fig, ax = plt.subplots(figsize=(5, 10))
    sns.heatmap(
        corr_series.to_frame(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        ax=ax,
    )
    ax.set_title("Correlation with Annual Salary")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Skill Analysis
# ---------------------------------------------------------------------------

def _get_skills_exploded(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["skills_list"] = temp["required_skills"].str.split("|")
    return temp.explode("skills_list").assign(
        skills_list=lambda x: x["skills_list"].str.strip()
    )


def plot_top_demand_vs_salary_skills(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Side-by-side: top N most in-demand skills vs avg salary per skill."""
    skills_exploded = _get_skills_exploded(df)
    result = (
        skills_exploded.groupby("skills_list")["annual_salary_usd"]
        .agg(["count", "mean"])
        .sort_values(by="count", ascending=False)
        .head(top_n)
        .round(2)
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    result["count"].plot(kind="barh", ax=axes[0], color="red")
    axes[0].set_title(f"Top {top_n} Most In-Demand Skills", fontsize=9)
    axes[0].set_xlabel("Number of Jobs", fontsize=8)
    axes[0].tick_params(labelsize=7)
    for i, v in enumerate(result["count"]):
        axes[0].text(v + 1, i, str(int(v)), va="center", fontsize=7)

    result["mean"].plot(kind="barh", ax=axes[1], color="orange")
    axes[1].set_title("Avg Salary by Skill", fontsize=9)
    axes[1].set_xlabel("Annual Salary (USD)", fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[1].set_xlim(180_000, 210_000)

    plt.tight_layout()
    return fig


def plot_high_paying_skills(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Side-by-side: top N highest-paying skills + their demand %."""
    skills_exploded = _get_skills_exploded(df)
    result = (
        skills_exploded.groupby("skills_list")["annual_salary_usd"]
        .agg(["count", "mean"])
        .sort_values(by="mean", ascending=False)
        .head(top_n)
        .round(2)
    )
    result_sorted = result.sort_values(by="mean", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    result_sorted["mean"].plot(kind="barh", color="green", ax=axes[0])
    axes[0].set_title(f"Top {top_n} High Paying Skills", fontsize=9)
    axes[0].set_xlabel("Annual Salary (USD)", fontsize=8)
    axes[0].set_ylabel("Skills", fontsize=8)
    axes[0].tick_params(labelsize=7)
    axes[0].set_xlim(220_000, 255_000)
    axes[0].grid(axis="x", linestyle="--", alpha=0.7)

    demand_pct = (result_sorted["count"] / len(df)) * 100
    demand_pct.plot(kind="barh", color="lightgreen", ax=axes[1])
    axes[1].set_title("Skill Demand (% of Total Jobs)", fontsize=9)
    axes[1].set_xlabel("Percentage (%)", fontsize=8)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelsize=7)
    axes[1].grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Job Category Overview
# ---------------------------------------------------------------------------

def plot_job_category_counts(df: pd.DataFrame) -> plt.Figure:
    """Count plot: all job categories sorted by frequency."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=df,
        x="job_category",
        color="midnightblue",
        order=df["job_category"].value_counts().index,
        ax=ax,
    )
    ax.set_title("Job Count by Category")
    ax.set_xlabel("Job Category")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_job_category_vs_industry(df: pd.DataFrame) -> plt.Figure:
    """Heatmap: job category distribution across industries (%)."""
    ct = round(pd.crosstab(df["job_category"], df["industry"], normalize="index") * 100, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(ct, annot=True, fmt=".1f", linewidths=0.5, cmap="PuOr", ax=ax)
    ax.set_title("Job Category × Industry Distribution (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. AI Engineering Deep Dive
# ---------------------------------------------------------------------------

def _get_ai_engineering_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["job_category"] == "AI Engineering"].copy()


def plot_ai_eng_salary_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap: AI Engineering job titles × experience level salary matrix."""
    temp_df = _get_ai_engineering_df(df)
    col_order = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"]

    pivot_df = (
        temp_df.pivot_table(
            index="job_title",
            columns="experience_level",
            values="annual_salary_usd",
            aggfunc="mean",
        )
        .reindex(columns=col_order)
        .round(2)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot_df, annot=True, cmap="rocket", fmt=".0f", ax=ax)
    ax.set_title("AI Engineering: Salary by Job Title & Experience Level")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_ai_eng_demand_metrics(df: pd.DataFrame) -> plt.Figure:
    """2×2 grid: demand score, growth YoY, benefits score, AI premium by AI Eng title."""
    temp_df = _get_ai_engineering_df(df)
    metrics = ["demand_score", "demand_growth_yoy_pct", "benefits_score_10", "ai_salary_premium_pct"]
    agg = temp_df.groupby("job_title")[metrics].mean().round(2)

    x = agg.sort_values("demand_score", ascending=False)
    y = agg.sort_values("demand_growth_yoy_pct", ascending=False)
    z = agg.sort_values("benefits_score_10", ascending=False)
    a = agg.sort_values("ai_salary_premium_pct", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    sns.barplot(data=x, x=x.index, y="demand_score", palette="viridis", ax=axes[0, 0])
    axes[0, 0].set_title("Avg Demand Score by Job Title")
    axes[0, 0].set_ylim(80, 100)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha="right")

    sns.barplot(data=y, x=y.index, y="demand_growth_yoy_pct", palette="RdBu", ax=axes[0, 1])
    axes[0, 1].set_title("Avg YoY Growth by Job Title")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha="right")

    sns.barplot(data=z, x=z.index, y="benefits_score_10", palette="magma", ax=axes[1, 0])
    axes[1, 0].set_title("Benefits Score by Job Title")
    axes[1, 0].set_ylim(7.5, 8.3)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha="right")

    sns.barplot(data=a, x=a.index, y="ai_salary_premium_pct", palette="Spectral", ax=axes[1, 1])
    axes[1, 1].set_title("AI Salary Premium (%)")
    axes[1, 1].set_ylim(5, 15)
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_ai_eng_education_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap: AI Engineering job title × education required (%)."""
    temp_df = _get_ai_engineering_df(df)
    ct = (
        pd.crosstab(temp_df["job_title"], temp_df["education_required"], normalize="index") * 100
    ).round(2)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(ct, annot=True, linewidths=0.5, cmap="YlOrRd", ax=ax)
    ax.set_title("AI Engineering: Education Required by Job Title (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Category × Avg Salary
# ---------------------------------------------------------------------------

def plot_category_avg_salary(df: pd.DataFrame) -> plt.Figure:
    """Bar chart: avg salary and avg experience per job category."""
    data = (
        df.groupby("job_category")[["years_of_experience", "annual_salary_usd"]]
        .mean()
        .round(2)
        .sort_values(by="annual_salary_usd", ascending=False)
        .reset_index().head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x="job_category", y="annual_salary_usd", palette="viridis", ax=ax)
    ax.set_title("Avg Salary by Job Category")
    ax.set_xlabel("Job Category")
    ax.set_ylabel("Avg Annual Salary (USD)")
    ax.set_ylim(160000,260000)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


## ---------------------------------------------------------------------------
# 10. Dynamic Category Analysis (MEDIUM SIZED PLOTS)
# ---------------------------------------------------------------------------

def _get_category_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Helper: filter dataframe by job category."""
    if category is None or category == "All" or category == "":
        return df.copy()
    return df[df["job_category"] == category].copy()


def plot_category_skill_demand(df: pd.DataFrame, category: str = "All") -> plt.Figure:
    """Horizontal bar: top 6 skills by % of job postings for selected category."""
    temp_df = _get_category_df(df, category)
    
    if len(temp_df) == 0:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5, f"No data for '{category}'", ha="center", va="center", fontsize=12)
        ax.set_title(f"Top Skills in {category} (No Data)")
        return fig
    
    ct = (
        temp_df["required_skills"]
        .str.split("|")
        .explode()
        .str.strip()
        .value_counts()
        .head(5)
    )
    pct = round(ct / len(temp_df) * 100, 2)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    pct.plot(kind="barh", colormap="coolwarm", ax=ax)
    ax.set_xlabel("% of Job Postings", fontsize=9)
    ax.set_title(f"Top 5 Skills in {category} Roles", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


def plot_category_salary_by_education(df: pd.DataFrame, category: str = "All") -> plt.Figure:
    """Heatmap: avg salary by education × job title for selected category."""
    temp_df = _get_category_df(df, category)
    
    if len(temp_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No data for '{category}'", ha="center", va="center", fontsize=12)
        ax.set_title(f"Salary by Education for {category}")
        return fig
    
    salary_pivot = temp_df.pivot_table(
        index="education_required",
        columns="job_title",
        values="annual_salary_usd",
        aggfunc="mean",
    ).round(0)
    
    # Limit columns for readability
    if len(salary_pivot.columns) > 8:
        top_cols = temp_df["job_title"].value_counts().head(8).index.tolist()
        salary_pivot = salary_pivot[top_cols]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(salary_pivot, annot=True, fmt=".0f", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title(f"Avg Salary by Education & Job Title ({category})", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_category_experience_transition(df: pd.DataFrame, category: str = "All") -> plt.Figure:
    """Grouped bar: experience_level × job title for selected category."""
    temp_df = _get_category_df(df, category)
    
    if len(temp_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No data for '{category}'", ha="center", va="center", fontsize=12)
        ax.set_title(f"Experience Distribution for {category}")
        return fig
    
    # Limit to top titles
    top_titles = temp_df["job_title"].value_counts().head(6).index.tolist()
    plot_df = temp_df[temp_df["job_title"].isin(top_titles)]
    
    z = pd.crosstab(plot_df["experience_level"], plot_df["job_title"])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    z.plot(kind="bar", ax=ax)
    ax.set_title(f"Job Title Distribution by Experience Level ({category})", fontsize=10)
    ax.set_xlabel("Experience Level", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(title="Job Title", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    ax.tick_params(labelsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_category_summary_stats(df: pd.DataFrame, category: str = "All") -> dict:
    """Return summary stats for selected category."""
    temp_df = _get_category_df(df, category)
    
    if len(temp_df) == 0:
        return {
            "count": 0,
            "avg_salary": 0,
            "top_titles": {},
            "ai_premium": 0,
            "benefits": 0,
            "demand_score": 0,
            "growth": 0,
        }
    
    return {
        "count": len(temp_df),
        "avg_salary": round(temp_df["annual_salary_usd"].mean(), 2),
        "top_titles": temp_df["job_title"].value_counts().head(5).to_dict(),
        "ai_premium": round(temp_df["ai_salary_premium_pct"].mean(), 2),
        "benefits": round(temp_df["benefits_score_10"].mean(), 2),
        "demand_score": round(temp_df["demand_score"].mean(), 2),
        "growth": round(temp_df["demand_growth_yoy_pct"].mean(), 2),
    }


def get_all_categories(df: pd.DataFrame) -> list:
    """Return sorted list of all job categories with 'All' as first option."""
    categories = sorted(df["job_category"].dropna().unique().tolist())
    return ["All"] + categories