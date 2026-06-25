import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset():
   df = pd.read_csv("../Salary/data/cleaned_data.csv")
   return df

def build_profiles(df):
    available = list(df.columns[26:])
    
    # Keep only numeric/binary columns (skill columns should be 0/1)
    available = [c for c in available if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    print("Skill columns used:", available)  # verify correct columns
    
    job_profiles = df.groupby("job_title")[available].mean()
    return job_profiles

def build_sim_matrix(job_profiles):
    sim_matrix = cosine_similarity(job_profiles)
    return pd.DataFrame(sim_matrix,
                        index=job_profiles.index,
                        columns=job_profiles.index)

def recommend(job_title, df, job_profiles, sim_df, top_n_skills=5, top_n_similar=3):
    if job_title not in job_profiles.index:
        matches = [t for t in job_profiles.index if job_title.lower() in t.lower()]
        if matches:
            job_title = matches[0]
        else:
            return None

    profile    = job_profiles.loc[job_title]
    top_skills = profile[profile > 0].sort_values(ascending=False).head(top_n_skills)

    similar    = sim_df[job_title].sort_values(ascending=False)
    similar    = similar[similar.index != job_title].head(top_n_similar)

    avg_salary = df[df["job_title"] == job_title]["annual_salary_usd"].mean()
    postings   = len(df[df["job_title"] == job_title])

    return {
        "job_title":   job_title,
        "avg_salary":  avg_salary,
        "postings":    postings,
        "top_skills":  top_skills,
        "similar_jobs": similar
    }
df           = load_dataset()
job_profiles = build_profiles(df)        # ← missing this line
sim_df       = build_sim_matrix(job_profiles)

# Test
result = recommend("Data Scientist", df, job_profiles, sim_df)

if result:
    print(f"Job Title  : {result['job_title']}")
    print(f"Avg Salary : ${result['avg_salary']:,.0f}")
    print(f"Postings   : {result['postings']}")
    print(f"\nTop Skills :\n{result['top_skills']}")
    print(f"\nSimilar Jobs:\n{result['similar_jobs']}")
else:
    print("Job title not found.")