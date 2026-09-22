"""
Entity extraction for resume screening.

Uses a blank spaCy pipeline with:
  1. EntityRuler with curated patterns for SKILL, JOB_TITLE, DEGREE entities
     (rule-based — no training data required), plus
  2. spaCy's statistical NER (en_core_web_sm, when available) for
     PERSON / ORG / GPE entities.

If en_core_web_sm is not installed, the pipeline still runs with the
rule-based EntityRuler only. Install it with:
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re

import spacy
from spacy.language import Language

SKILLS = [
    "SQL", "Excel", "Power BI", "Tableau", "Looker", "DAX", "Power Query",
    "Python", "R", "machine learning", "statistical modeling",
    "hypothesis testing", "regression", "scikit-learn", "pandas", "NumPy",
    "experimental design", "feature engineering", "model evaluation",
    "TensorFlow", "PyTorch", "MLflow", "Docker", "Kubernetes",
    "model deployment", "MLOps", "XGBoost", "deep learning", "REST APIs",
    "CI/CD", "Spark", "feature stores", "SSRS", "data warehousing",
    "dimensional modeling", "dashboard design", "ETL", "data governance",
    "business requirements", "stakeholder management", "Airflow", "dbt",
    "Kafka", "data pipelines", "AWS", "GCP", "Snowflake", "Databricks",
    "schema design", "data quality", "data validation", "pivot tables",
    "statistical analysis", "A/B testing", "reporting", "data cleaning",
    "KPIs", "stakeholder reporting",
]

JOB_TITLES = [
    "Data Analyst", "Senior Data Analyst", "Analytics Associate",
    "Data Scientist", "Senior Data Scientist", "Research Analyst",
    "Machine Learning Engineer", "ML Engineer", "Applied ML Engineer",
    "BI Analyst", "Business Intelligence Analyst", "BI Developer",
    "Data Engineer", "Senior Data Engineer", "Analytics Engineer",
]

DEGREES = [
    "Bachelor of Science", "Bachelor of Technology", "B.S.", "B.Tech",
    "Master of Science", "M.S.", "MBA", "PhD", "Ph.D.",
]


def _patterns_for(values: list[str], label: str) -> list[dict]:
    patterns = []
    for v in values:
        token_patterns = [{"LOWER": t.lower()} for t in v.split()]
        patterns.append({"label": label, "pattern": token_patterns})
    return patterns


def build_pipeline() -> Language:
    """Build the NER pipeline: EntityRuler + statistical NER when available."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")

    ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)
    patterns = (
        _patterns_for(SKILLS, "SKILL")
        + _patterns_for(JOB_TITLES, "JOB_TITLE")
        + _patterns_for(DEGREES, "DEGREE")
    )
    ruler.add_patterns(patterns)
    return nlp


def extract_entities(text: str, nlp: Language) -> dict:
    """Extract structured entities from a resume."""
    doc = nlp(text)
    entities = {"SKILL": [], "JOB_TITLE": [], "DEGREE": [], "PERSON": [], "ORG": []}
    seen = {k: set() for k in entities}
    for ent in doc.ents:
        if ent.label_ in entities and ent.text not in seen[ent.label_]:
            entities[ent.label_].append(ent.text)
            seen[ent.label_].add(ent.text)

    # Experience spans: "N years of experience"
    years = re.findall(r"(\d+)\s+years?\s+of\s+experience", text, flags=re.I)
    entities["YEARS_OF_EXPERIENCE"] = [int(y) for y in years]
    return entities


if __name__ == "__main__":
    import pprint

    nlp = build_pipeline()
    demo = (
        "Senior Data Analyst with 4 years of experience at InsightWorks. "
        "Skills: SQL, Python, Power BI, DAX. Bachelor of Science from State University."
    )
    import pprint

    pprint.pprint(extract_entities(demo, nlp))
