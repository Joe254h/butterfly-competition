import json
import os
import sys
from datetime import datetime

LEADERBOARD_FILE = "leaderboard/README.md"
SCORES_FILE      = "scores.json"


def load_scores():
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            return json.load(f)
    return []


def save_scores(scores):
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)


def update_leaderboard(accuracy, f1, username):
    scores = load_scores()

    scores.append({
        "user":     username,
        "accuracy": accuracy,
        "f1":       f1,
        "date":     datetime.utcnow().strftime("%Y-%m-%d"),
    })

    save_scores(scores)

    best = {}
    for s in scores:
        u = s["user"]
        if u not in best or s["accuracy"] > best[u]["accuracy"]:
            best[u] = s

    top20  = sorted(best.values(), key=lambda x: x["accuracy"], reverse=True)[:20]
    medals = {1: "1st", 2: "2nd", 3: "3rd"}

    lines = [
        "# Butterfly Classification Leaderboard",
        "",
        f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "| Rank | Participant | Accuracy | F1 (macro) | Date |",
        "|------|-------------|----------|------------|------|",
    ]

    for i, s in enumerate(top20):
        rank  = i + 1
        label = medals.get(rank, str(rank))
        acc   = f"{s['accuracy']*100:.2f}%"
        f1_val = f"{s['f1']*100:.2f}%"
        lines.append(f"| {label} | **{s['user']}** | {acc} | {f1_val} | {s['date']} |")

    lines += [
        "",
        f"*Best score per participant. Total submissions: {len(scores)}*",
    ]

    os.makedirs("leaderboard", exist_ok=True)
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("\n".join(lines))

    rank_pos = next((i+1 for i, s in enumerate(top20) if s["user"] == username), "N/A")
    print(f"Leaderboard updated. {username}: {accuracy*100:.2f}% — Rank #{rank_pos}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leaderboard/update.py score.json")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = json.load(f)
    username = os.environ.get("GITHUB_ACTOR", "unknown")
    update_leaderboard(data["accuracy"], data.get("f1_macro", 0.0), username)
```

---

## `requirements.txt`
```
numpy
pandas
scikit-learn
