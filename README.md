<img width="1600" height="1066" alt="image" src="https://github.com/user-attachments/assets/1a19df1b-5506-4fb4-a93f-4777813570ac" />


# 🦋 Butterfly Image Classification Competition

**Goal:** Build the best butterfly species classifier.  
**Metric:** Accuracy + F1 (macro)  
**Classes:** 75 butterfly species  
**Live Leaderboard:** [leaderboard/README.md](leaderboard/README.md)

---

## 📁 Repository Structure
```
butterfly-competition/
├── .github/workflows/main.yml    # Auto-evaluation — DO NOT TOUCH
├── evaluation/score.py           # Scoring script — DO NOT TOUCH
├── leaderboard/
│   ├── README.md                 # Live rankings
│   └── update.py                 # Leaderboard updater — DO NOT TOUCH
├── submissions/                  # PUT YOUR SUBMISSION FILE HERE
├── requirements.txt              # Dependencies
└── scores.json                   # All scores database
```

---

## 🚀 How to Participate

### Step 1 — Open the Colab Notebook

Click the link below to open the competition notebook:

👉 **[Open Colab Notebook](YOUR_COLAB_LINK_HERE)**

The notebook will:
- Download the dataset automatically from Kaggle
- Train an EfficientNetV2 baseline model
- Generate your `submission.csv`
- Give you a download button for your predictions

---

### Step 2 — Run All Cells

Run every cell from top to bottom.  
In the last cell set your name:
```python
YOUR_NAME = "your_name_here"
```

Then click the green **Download** button that appears.

---

### Step 3 — Submit to GitHub

1. Go to [submissions/](https://github.com/Joe254h/butterfly-competition/tree/main/submissions) folder
2. Click **Add file → Upload files**
3. Upload your `YOUR_NAME_submission.csv`
4. At the bottom select **"Create a new branch"** — NOT commit to main
5. Click **Propose changes**
6. Click **Create pull request**

GitHub automatically evaluates your submission and posts your score as a comment on the PR.

---

## 🏆 Leaderboard

Rankings update automatically after every submission.

👉 [View Live Leaderboard](leaderboard/README.md)

---

## 📏 Evaluation

Your submission is scored on **1300 hidden test images** using:

| Metric | Description |
|--------|-------------|
| Accuracy | Correct predictions / total predictions |
| F1 (macro) | Average F1 across all 75 species |

Your `submission.csv` must have:
- One species name per row
- No header
- Exactly 1300 predictions
- Labels matching exactly the 75 class names in the dataset

Example:
```
MONARCH
PAPER KITE
BLUE MORPHO
RED ADMIRAL
...
```

---

## 💡 Ideas to Beat the Baseline

| Strategy | Expected Gain |
|----------|--------------|
| Unfreeze EfficientNet layers | +5–10% |
| Add data augmentation | +3–7% |
| Train more epochs | +2–5% |
| Use EfficientNetV2-L | +3–8% |
| Ensemble multiple models | +5–10% |

---

## 📋 Rules

1. One `.csv` file per Pull Request
2. File must go in `submissions/` folder
3. One predicted species name per row, no header
4. Exactly 1300 predictions
5. Do not modify any files outside `submissions/`
6. Do not share or leak test labels

---

## ❓ Common Issues

**My PR workflow failed** — Check that your file is in `submissions/` and you opened a PR not a direct commit to main.

**Wrong number of predictions** — Your submission must have exactly 1300 rows. Re-run Cell 12 and Cell 13 in the notebook.

**Invalid species names** — Labels must match exactly. Check the notebook Cell 12 sanity check output.
