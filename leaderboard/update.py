import json
import os
import sys
from datetime import datetime

LEADERBOARD_FILE = "leaderboard/README.md"
SCORES_FILE = "scores.json"


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

    submission = {
        "user": username,
        "accuracy": accuracy,
        "f1": f1,
        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    }

    scores.append(submission)

    save_scores(scores)

    # keep best score per user
    best_scores = {}

    for s in scores:
        user = s["user"]

        if user not in best_scores or s["accuracy"] > best_scores[user]["accuracy"]:
            best_scores[user] = s

    leaderboard = sorted(
        best_scores.values(),
        key=lambda x: x["accuracy"],
        reverse=True
    )

    medals = ["🥇", "🥈", "🥉"]

    lines = [
        "# 🦋 Butterfly Classification Leaderboard",
        "",
        f"Last updated: **{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}**",
        "",
        "| Rank | Participant | Accuracy | F1 (macro) | Date |",
        "|------|-------------|----------|------------|------|"
    ]

    for i, s in enumerate(leaderboard[:20]):

        rank = medals[i] if i < 3 else str(i + 1)

        acc = f"{s['accuracy']*100:.2f}%"
        f1_val = f"{s['f1']*100:.2f}%"

        lines.append(
            f"| {rank} | **{s['user']}** | {acc} | {f1_val} | {s['date']} |"
        )

    lines.append("")
    lines.append(f"**Total submissions:** {len(scores)}")
    lines.append(f"**Participants:** {len(best_scores)}")

    os.makedirs("leaderboard", exist_ok=True)

    with open(LEADERBOARD_FILE, "w") as f:
        f.write("\n".join(lines))

    print("Leaderboard updated successfully.")


if __name__ == "__main__":

    score_file = "score.json"

    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        score_file = sys.argv[1]

    if not os.path.exists(score_file):
        print("score.json not found.")
        sys.exit(1)

    with open(score_file) as f:
        data = json.load(f)

    username = os.environ.get("GITHUB_ACTOR", "local_user")

    update_leaderboard(
        data["accuracy"],
        data.get("f1_macro", 0.0),
        username
    )
