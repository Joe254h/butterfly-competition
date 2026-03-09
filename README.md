🦋 Butterfly Classification Competition

> A mini Kaggle-style competition — classify **75 butterfly species** and climb the leaderboard automatically!

## 🎯 Goal
Build the best image classifier. **Metric: Accuracy**

## 📁 Structure
```
butterfly-competition/
├── .github/workflows/evaluate.yml   # ⚙️ Auto-evaluation (DO NOT TOUCH)
├── baseline/model.py                # 🦋 Baseline model to improve
├── data/
│   ├── train/                       # Training images
│   ├── test/                        # Test images
│   └── Training_set.csv             # Filenames + labels
├── evaluation/score.py              # 📊 Scoring (DO NOT TOUCH)
├── leaderboard/README.md            # 🏆 LIVE RANKINGS
├── submissions/                     # 📤 PUT YOUR FILE HERE
├── train.py                         # Train the model
├── predict.py                       # Generate predictions
└── requirements.txt
```

## 🚀 Step-by-Step Guide

### 1 — Get the Data
Download from Kaggle: https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species

Put in `data/` so you have `data/train/`, `data/test/`, `data/Training_set.csv`.

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Train
```bash
python train.py
```
Saves `model_weights.h5` and `label_classes.npy`.

### 4 — Generate predictions
```bash
python predict.py
```
Creates `submissions/submission.csv`.

### 5 — Submit
```bash
mv submissions/submission.csv submissions/YOUR_NAME_submission.csv
git add submissions/YOUR_NAME_submission.csv
git commit -m "My submission"
git push
```
Then open a **Pull Request** on GitHub. GitHub scores it automatically!

## 🏆 Leaderboard
See rankings → [leaderboard/README.md](leaderboard/README.md)

## 💡 Ideas to Beat the Baseline
- Fine-tune EfficientNet (`trainable=True`)
- Add data augmentation (flips, rotations, brightness)
- Try a larger EfficientNetV2 variant
- Train more epochs with early stopping
- Use an ensemble of models

## 📋 Submission Rules
1. One `.csv` file per Pull Request in `submissions/`
2. One predicted label per row, no header
3. Labels must match exactly the 75 class names in `Training_set.csv`
