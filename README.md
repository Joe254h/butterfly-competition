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

👉 **[Open Colab Notebook](https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/joelnyong/butterfly-competition.84090f66-6fe2-4fc1-82a3-20da36430354.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20260310/auto/storage/goog4_request%26X-Goog-Date%3D20260310T104332Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D491fc9f53bd2c57673f471e5690d8481b1cc484963fd3c6d19ace354748e99d601426644d1af159393ce002146055141da6b3e1985445420bd5e15ab6807a86241f238596ce972562331c105e7806340c8ae7606b1ac38d7c34062fccffa3380a82235d4f7b91fb20be4c578c722c79036b9d51b5027b7bafce958b68904f45a60e79342a17c54b73276441a3b5f896fb8bb191c6bdedfcfecbfef4b40365e320fb80d8fab0ed9a8167529c9c79fac5b390a771837a4e5c2a320dbb11eb307fc4ea0fd64fb87149e162e4171f18a5cbc0342f94447ec8106638993928b9b845906ca275f1de6ab2f41976fcfd00d3e38ee5745840fbc82a981779310a3ee3f91)**

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
