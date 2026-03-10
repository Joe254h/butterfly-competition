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

    # Keep best score per participant
    best = {}
    for s in scores:
        u = s["user"]
        if u not in best or s["accuracy"] > best[u]["accuracy"]:
            best[u] = s

    # Sort all participants by accuracy
    ranked = sorted(best.values(), key=lambda x: x["accuracy"], reverse=True)

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}

    lines = [
        "# 🦋 Butterfly Classification Leaderboard",
        "",
        f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        f"**Total submissions:** {len(scores)} &nbsp;|&nbsp; **Participants:** {len(ranked)}",
        "",
        "| Rank | Participant | Accuracy | F1 (macro) | Date |",
        "|------|-------------|----------|------------|------|",
    ]

    for i, s in enumerate(ranked):
        rank   = i + 1
        medal  = medals.get(rank, str(rank))
        acc    = f"{s['accuracy']*100:.2f}%"
        f1_val = f"{s['f1']*100:.2f}%"
        lines.append(
            f"| {medal} | **{s['user']}** | {acc} | {f1_val} | {s['date']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 📤 How to Submit",
        "",
        "1. Open the Colab notebook and run all cells",
        "2. Download your `YOUR_NAME_submission.csv`",
        "3. Go to [submissions/](../submissions/) → **Add file → Upload files**",
        "4. Select **Create a new branch** at the bottom — NOT commit to main",
        "5. Click **Propose changes → Create pull request**",
        "6. Your score appears here automatically",
        "",
        "---",
        "",
        f"*Best score per participant shown. Total submissions: {len(scores)}*",
    ]

    os.makedirs("leaderboard", exist_ok=True)
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("\n".join(lines))

    rank_pos = next((i+1 for i, s in enumerate(ranked) if s["user"] == username), "N/A")
    print(f"Leaderboard updated. {username}: {accuracy*100:.2f}% — Rank #{rank_pos} of {len(ranked)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leaderboard/update.py score.json")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = json.load(f)
    username = os.environ.get("GITHUB_ACTOR", "unknown")
    update_leaderboard(data["accuracy"], data.get("f1_macro", 0.0), username)
