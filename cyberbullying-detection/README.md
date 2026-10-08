# AI-Based Cyberbullying Comment Detection

A machine-learning pipeline that classifies social-media comments as **bullying** or **not bullying**.

**Pipeline:** text cleaning (lower-casing, leetspeak normalisation, URL/mention removal) -> TF-IDF features (word 1-2 grams + character 3-5 grams) -> classifier (Naive Bayes, Logistic Regression, Linear SVM) -> evaluation -> saved best model.

## Run
```bash
pip install -r requirements.txt
python src/make_dataset.py   # builds data/comments.csv (synthetic demo data)
python src/train.py          # trains, evaluates, saves models/best_model.joblib + results/*.png
python src/predict.py "you are such a loser"   # classify your own comment
```

## Results (demo data)
See `results/metrics.json`, `results/run_log.txt`, `results/confusion_matrix.png`, `results/model_comparison.png`.

## Important note on data
`data/comments.csv` is **synthetic** (template-generated), so scores are near-perfect and do **not** reflect real-world performance. For real results, replace it with a real labelled dataset (e.g. Kaggle "Cyberbullying Classification" or Jigsaw Toxic Comment) using columns `text,label` (plus an optional `group` column), then re-run `train.py`.

## Structure
```
src/make_dataset.py  src/train.py  src/predict.py
data/comments.csv    models/best_model.joblib   results/
```
Author: Ayisha Shana Perumballi
