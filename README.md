# Automated Resume Screening & Classification

An NLP pipeline that reads resumes, extracts structured information, classifies
candidates into role categories, and scores them against job descriptions.

## How it works

1. **Synthetic data** — `src/generate_data.py` creates 300 labeled resumes across
   5 role categories plus job descriptions. All names, employers, schools and
   contact details are fictitious.
2. **Entity extraction** — `src/entity_extraction.py` builds a spaCy pipeline that
   combines:
   - a rule-based `EntityRuler` with curated patterns for `SKILL`, `JOB_TITLE`
     and `DEGREE` entities (works with zero training data), and
   - spaCy's statistical NER (`en_core_web_sm`, used if installed) for
     `PERSON` / `ORG` entities.
   - a regex layer for years-of-experience spans.
3. **Classification** — `src/train_classifier.py` trains a
   TF-IDF (unigrams + bigrams) → `LogisticRegression` pipeline on a stratified
   80/20 split and reports accuracy and a classification report on a held-out
   eval set.
4. **Screening demo** — `src/demo.py` loads the trained model, extracts skills
   from a resume and a job description, and prints a screening verdict based on
   skill-overlap match.

## Results

- Held-out eval accuracy: **1.00** on 60 test resumes (5 role categories;
  synthetic resumes have strongly separated skill vocabularies, so a linear
  TF-IDF model separates them cleanly).
- Screening score = fraction of required job-description skills found in the
  resume; verdict thresholds: ≥60% strong match, ≥30% moderate, else weak.

## How to run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional; rule-based NER works without it
python src/generate_data.py
python src/train_classifier.py
python src/demo.py
```

## Project structure

```
automated-resume-screening/
├── src/
│   ├── generate_data.py      # synthetic resume + JD generator
│   ├── entity_extraction.py  # spaCy EntityRuler + statistical NER
│   ├── train_classifier.py   # TF-IDF + LogisticRegression training
│   └── demo.py               # end-to-end screening demo
├── data/                     # generated CSVs (git-ignored; regenerate locally)
├── models/                   # trained classifier artifact (git-ignored)
└── notebooks/                # exploratory analysis
```

## Notes

- The dataset is fully synthetic; no real personal data is used.
- In production this design generalizes to a fine-tuned transformer classifier
  and a larger, human-labeled corpus.
