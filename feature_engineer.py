import pandas as pd

# 1. Load data
df = pd.read_csv("../Salary/data/ai_jobs_market_2025_2026.csv")

# 2. Extract and find top 93 skills (excluding Linux)
df["skills_list"] = df["required_skills"].str.split("|")
skills_exploded = df.explode("skills_list")
skills_exploded["skills_list"] = skills_exploded["skills_list"].str.strip()

top_skills = (
    skills_exploded["skills_list"].value_counts().head(93).index.to_list()
)
if "Linux" in top_skills:
    top_skills.remove("Linux")

# 3. One-hot encode individual skills
for skill in top_skills:
    df[skill] = (
        df["required_skills"]
        .str.contains(skill, na=False, regex=False)
    ).astype(int)

# 4. Merge variations using Regex mappings
merges = {
    "Cloud": r"Cloud|Cloud \(AWS/GCP/Azure\)",  # escaped parenthesis
    "System_Design": r"System Design|System_Design",
    "RAG": r"RAG|Rag",
    "MLOps": r"MLOps|Mlops",
    "Fine-Tuning": r"Fine-tuning|Fine_Tune|LLM Fine-tuning",
    "LLM_APIs": r"LLM APIs|LLM_APIs|LLM Integration|GenAI APIs",
    "Distributed_Systems": r"Distributed Systems|Distributed_Systems",
    "Prompt_Engineering": r"Prompt Engineering|Prompt Design",
    "Risk Analysis": r"Risk Management|Risk Analysis|Risk Assessment",
}

for new_col, pattern in merges.items():
    df[new_col] = (
        df["required_skills"].str.contains(pattern, na=False, regex=True)
    ).astype(int)

# 5. Drop old redundant columns
columns_to_drop = [
    "skills_list",
    "Cloud (AWS/GCP/Azure)",
    "LLM Fine-tuning",
    "LLM Integration",
    "GenAI APIs",
    "LLM APIs",
    "Prompt Design",
    "Risk Management",
    "Risk Assessment",
    "System Design",
    "Rag",
    "Mlops",
    "Fine_Tune",
]
df = df.drop(columns=columns_to_drop, errors="ignore")

# 6. Save back to a new CSV
df.to_csv("../Salary/data/cleaned_data.csv", index=False)
print("Feature engineering complete!")