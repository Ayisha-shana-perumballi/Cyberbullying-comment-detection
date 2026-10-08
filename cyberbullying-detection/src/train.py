"""Train & evaluate cyberbullying comment classifiers."""
import re, json, joblib, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

def clean(t):
    t = str(t).lower()
    t = t.translate(str.maketrans({"@":"a","1":"i","0":"o","$":"s","3":"e"}))
    t = re.sub(r"http\S+|@\w+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

df = pd.read_csv("data/comments.csv").dropna()
df["clean"] = df["text"].map(clean)
# Group split: whole templates are held out, so the test set contains sentence patterns never seen in training
gss = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=7)
tr, te = next(gss.split(df, df["label"], df["group"]))
X_tr, X_te = df["clean"].iloc[tr], df["clean"].iloc[te]
y_tr, y_te = df["label"].iloc[tr], df["label"].iloc[te]

def feats():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, sublinear_tf=True)),
    ])
models = {
    "Naive Bayes": Pipeline([("f", TfidfVectorizer(ngram_range=(1,2), min_df=2)), ("c", MultinomialNB())]),
    "Logistic Regression": Pipeline([("f", feats()), ("c", LogisticRegression(max_iter=1000, C=3))]),
    "Linear SVM": Pipeline([("f", feats()), ("c", LinearSVC(C=1))]),
}
res, fitted = {}, {}
cv = GroupKFold(n_splits=5)
for n, m in models.items():
    m.fit(X_tr, y_tr); p = m.predict(X_te); fitted[n] = (m, p)
    res[n] = dict(accuracy=accuracy_score(y_te,p), precision=precision_score(y_te,p),
                  recall=recall_score(y_te,p), f1=f1_score(y_te,p),
                  cv_f1_mean=cross_val_score(m, df["clean"], df["label"], cv=cv, groups=df["group"], scoring="f1").mean())
    print(n, {k: round(v,4) for k,v in res[n].items()})
best = max(res, key=lambda k: res[k]["f1"])
print("Best:", best)
print(classification_report(y_te, fitted[best][1], target_names=["Not bullying","Bullying"]))
json.dump({"results":res,"best":best,"train_size":len(X_tr),"test_size":len(X_te)}, open("results/metrics.json","w"), indent=2)
joblib.dump(fitted[best][0], "models/best_model.joblib")

cm = confusion_matrix(y_te, fitted[best][1])
ConfusionMatrixDisplay(cm, display_labels=["Not bullying","Bullying"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best}"); plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

names = list(res); x = np.arange(len(names)); w = .2
for i, k in enumerate(["accuracy","precision","recall","f1"]):
    plt.bar(x+i*w, [res[n][k] for n in names], w, label=k)
plt.xticks(x+1.5*w, names); plt.ylim(.5,1.05); plt.legend(); plt.title("Model comparison")
plt.tight_layout(); plt.savefig("results/model_comparison.png", dpi=150); plt.close()
