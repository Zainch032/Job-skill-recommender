from pathlib import Path
import pandas as pd

# 1. Dynamically get the directory where feature_engineer.py lives
BASE_DIR = Path(__file__).resolve().parent

# 2. Path to the data folder and file right next to it
DATA_PATH = BASE_DIR / "data" / "ai_jobs_market_2025_2026.csv"

def load_data():
    """
    Generically loads the dataset from the absolute path calculated at runtime.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find the dataset. Please ensure your folder structure matches. "
            f"Looked here: {DATA_PATH.resolve()}"
        )
    
    df = pd.read_csv(DATA_PATH)
    return df
df = load_data()
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
    "Distributed Systems",
    "Rag",
    "Mlops",
    "Fine_Tune",
]
df = df.drop(columns=columns_to_drop, errors="ignore")

# 6. Save back to a new CSV
OUTPUT_PATH = BASE_DIR / "data" / "cleaned_data.csv"
df.to_csv(OUTPUT_PATH, index=False)
print("Feature engineering complete!")