"""
Synthetic data generator for the resume screening project.

Creates:
  data/synthetic_resumes.csv  — labeled resume texts + role categories
  data/job_descriptions.csv   — job postings per role category

All names, employers, schools, emails and addresses are fictitious.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
random.seed(42)

ROLE_CATEGORIES = [
    "Data Analyst",
    "Data Scientist",
    "Machine Learning Engineer",
    "Business Intelligence Analyst",
    "Data Engineer",
]

FIRST_NAMES = [
    "Aarav", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Arjun", "Divya",
    "Karan", "Meera", "Rohit", "Ishita", "Sanjay", "Pooja", "Aditya", "Kavya",
    "Nikhil", "Ritu", "Varun", "Anika", "Farhan", "Lakshmi", "Manoj", "Neha",
]
LAST_NAMES = [
    "Sharma", "Patel", "Reddy", "Iyer", "Khan", "Gupta", "Mehta", "Nair",
    "Joshi", "Das", "Kulkarni", "Verma", "Singh", "Chandra", "Bose", "Agarwal",
]
EMPLOYERS = [
    "TechNova Solutions", "DataBridge Labs", "InsightWorks", "CloudPeak",
    "QuantumMetrics", "BrightSpark Analytics", "Nexalytics", "PrimeSight",
    "BlueOrbit Data", "ApexInsight",
]
SCHOOLS = [
    "State University", "Institute of Technology", "Metropolitan College",
    "National University", "Central State Institute",
]
SKILL_POOLS = {
    "Data Analyst": [
        "SQL", "Excel", "Power BI", "Tableau", "data validation", "pivot tables",
        "statistical analysis", "A/B testing", "reporting", "ETL", "data cleaning",
        "DAX", "Power Query", "stakeholder reporting", "KPIs",
    ],
    "Data Scientist": [
        "Python", "R", "machine learning", "statistical modeling", "hypothesis testing",
        "regression", "scikit-learn", "pandas", "NumPy", "experimental design",
        "feature engineering", "model evaluation", "SQL",
    ],
    "Machine Learning Engineer": [
        "Python", "TensorFlow", "PyTorch", "MLflow", "Docker", "Kubernetes",
        "model deployment", "MLOps", "XGBoost", "deep learning", "REST APIs",
        "CI/CD", "Spark", "feature stores", "A/B testing of models",
    ],
    "Business Intelligence Analyst": [
        "Power BI", "Tableau", "Looker", "DAX", "SQL", "data warehousing",
        "dimensional modeling", "SSRS", "dashboard design", "ETL", "data governance",
        "business requirements", "stakeholder management",
    ],
    "Data Engineer": [
        "Python", "SQL", "Spark", "Airflow", "dbt", "Kafka", "data pipelines",
        "AWS", "GCP", "data warehousing", "Snowflake", "ETL", "Databricks",
        "schema design", "data quality",
    ],
}
TITLE_POOLS = {
    "Data Analyst": ["Data Analyst", "Senior Data Analyst", "Analytics Associate"],
    "Data Scientist": ["Data Scientist", "Senior Data Scientist", "Research Analyst"],
    "Machine Learning Engineer": ["Machine Learning Engineer", "ML Engineer", "Applied ML Engineer"],
    "Business Intelligence Analyst": ["BI Analyst", "Business Intelligence Analyst", "BI Developer"],
    "Data Engineer": ["Data Engineer", "Senior Data Engineer", "Analytics Engineer"],
}


def _bullet(verbs, nouns) -> str:
    return f"{random.choice(verbs)} {random.choice(nouns)} using {random.choice(nouns)}"


def make_resume(role: str, idx: int) -> tuple[str, str]:
    """Return (resume_text, role). Resumes from other roles bleed in noise skills."""
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    email = re.sub(r"[^a-z]", "", name.lower()) + f"{idx}@examplemail.com"
    employer = random.choice(EMPLOYERS)
    school = random.choice(SCHOOLS)
    title = random.choice(TITLE_POOLS[role])

    skills = SKILL_POOLS[role][:]
    noise = [s for r, pool in SKILL_POOLS.items() if r != role for s in pool]
    sampled = random.sample(skills, k=random.randint(8, 12))
    sampled += random.sample(noise, k=random.randint(2, 5))
    random.shuffle(sampled)

    years = random.randint(1, 6)
    bullets = [
        f"Built and maintained dashboards tracking {random.randint(5, 30)} KPIs "
        f"for {random.randint(2, 6)} stakeholder teams.",
        f"Automated reporting workflows with {random.choice(['Python', 'SQL', 'Excel'])}, "
        f"saving {random.randint(5, 20)} hours per week.",
        f"Improved data quality by {random.randint(10, 40)}% through validation checks "
        f"and anomaly detection.",
    ]
    text = f"""
{name}
{email} | (555) 010-{idx:04d}

SUMMARY
{title} with {years} years of experience at {employer}, specializing in {role.lower()} work.

SKILLS
{', '.join(sampled)}

EXPERIENCE
{title} | {employer} | 20{18 + random.randint(0, 6)}-Present
- {bullets[0]}
- {bullets[1]}
- {bullets[2]}

EDUCATION
Bachelor of Science | {school} | 20{14 + random.randint(0, 6)}
"""
    return text.strip(), role


def make_job_description(role: str) -> tuple[str, str]:
    skills = random.sample(SKILL_POOLS[role], k=8)
    text = f"""
{' '.join(role.split())} — {random.choice(EMPLOYERS)}

We are hiring a {role} to join our growing analytics team.

Responsibilities:
- Design, build, and maintain scalable analytics solutions.
- Partner with business stakeholders to define KPIs and success metrics.
- Ensure data quality, documentation, and reproducibility of analyses.

Required skills: {', '.join(skills)}.
"""
    return text.strip(), role


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for role in ROLE_CATEGORIES:
        for _ in range(60):
            text, label = make_resume(role, idx)
            rows.append({"resume_text": text, "role_category": label})
            idx += 1
    pd.DataFrame(rows).to_csv(DATA_DIR / "synthetic_resumes.csv", index=False)

    jobs = [make_job_description(role) for role in ROLE_CATEGORIES for _ in range(3)]
    pd.DataFrame(jobs, columns=["job_description", "role_category"]).to_csv(
        DATA_DIR / "job_descriptions.csv", index=False
    )
    print(f"Wrote {len(rows)} resumes and {len(jobs)} job descriptions to {DATA_DIR}")


if __name__ == "__main__":
    main()
