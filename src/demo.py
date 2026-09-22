"""
Demo: screen a sample resume against a sample job description.

Loads the trained classifier, extracts entities with spaCy, computes a
skill-overlap match score between the resume and a job description, and
prints a short screening summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from entity_extraction import build_pipeline, extract_entities

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "resume_classifier.joblib"
JOBS_PATH = BASE_DIR / "data" / "job_descriptions.csv"


def match_score(resume_skills: list[str], job_skills: list[str]) -> float:
    job = {s.lower() for s in job_skills}
    if not job:
        return 0.0
    matched = {s.lower() for s in resume_skills} & job
    return len(matched) / len(job)


def main() -> None:
    resumes = pd.read_csv(BASE_DIR / "data" / "synthetic_resumes.csv")
    jobs = pd.read_csv(JOBS_PATH)

    clf = joblib.load(MODEL_PATH)
    nlp = build_pipeline()

    resume_text = resumes["resume_text"].iloc[0]
    job_row = jobs[jobs["role_category"] == "Data Analyst"].iloc[0]

    predicted_role = clf.predict([resume_text])[0]
    probs = clf.predict_proba([resume_text])[0]
    top3 = sorted(zip(clf.classes_, probs), key=lambda x: x[1], reverse=True)[:3]

    resume_ents = extract_entities(resume_text, nlp)
    job_ents = extract_entities(job_row["job_description"], nlp)
    score = match_score(resume_ents["SKILL"], job_ents["SKILL"])

    print("=" * 60)
    print("RESUME SCREENING RESULT")
    print("=" * 60)
    print(f"Predicted role : {predicted_role}")
    print("Top-3 class probabilities:")
    for cls, p in top3:
        print(f"  - {cls}: {p:.2f}")
    print(f"Skills found   : {len(resume_ents['SKILL'])} -> {', '.join(resume_ents['SKILL'][:8])}")
    print(f"Job titles     : {', '.join(resume_ents['JOB_TITLE'])}")
    print(f"Experience     : {resume_ents['YEARS_OF_EXPERIENCE']} years (extracted)")
    print(f"Target job     : {job_row['role_category']}")
    print(f"Skill match    : {score:.0%} of required skills present")
    verdict = (
        "STRONG MATCH — advance to interview screen"
        if score >= 0.6
        else "MODERATE MATCH — review manually"
        if score >= 0.3
        else "WEAK MATCH — screen out"
    )
    print(f"Screening verdict: {verdict}")


if __name__ == "__main__":
    main()
