"""Usage: python src/predict.py "your comment here"   (no args = demo)"""
import sys, re, joblib

def clean(t):
    t = str(t).lower()
    t = t.translate(str.maketrans({"@":"a","1":"i","0":"o","$":"s","3":"e"}))
    t = re.sub(r"http\S+|@\w+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

model = joblib.load("models/best_model.joblib")
samples = sys.argv[1:] or [
    "You are such an idiot, nobody likes you",
    "Great work on the project, really inspiring!",
    "you killed it in that performance!",
    "go away you ugly loser",
    "Can anyone help me with this assignment?",
]
for s in samples:
    p = model.predict([clean(s)])[0]
    print(f"[{'BULLYING' if p else 'SAFE    '}] {s}")
