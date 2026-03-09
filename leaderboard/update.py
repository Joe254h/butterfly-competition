
import json, os, sys
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


def update_leaderboard(new_score, username):
    scores = load_scores()

    # Append new submission
    scores.append({
        "user":  username,
        "score": new_score,
        "date":  datetime.utcnow().strftime("%Y-%m-%d"),
    })

    # Keep full history sorted
    scores.sort(key=lambda x: x["score"], reverse=True)
    save_scores(scores)

    # Best score per user only (for display)
    best = {}
    for s in scores:
        u = s["user"]
        if u not in best or s["score"] > best[u]["score"]:
            best[u] = s

    top20 = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:20]

    # Build Markdown
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    lines = [
        "# 🦋 Butterfly Classification Leaderboard\n",
        f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n",
        "| Rank | Participant | Accuracy | Date |",
        "|------|-------------|----------|------|",
    ]
    for i, s in enumerate(top20):
        rank   = i + 1
        medal  = medals.get(rank, "")
        acc    = f"{s['score']*100:.2f}%"
        lines.append(f"| {medal} {rank} | **{s['user']}** | {acc} | {s['date']} |")

    lines += [
        "",
        f"*Best score per participant shown. Total submissions: {len(scores)}*",
    ]

    os.makedirs("leaderboard", exist_ok=True)
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("\n".join(lines))

    rank_pos = next((i+1 for i, s in enumerate(top20) if s["user"] == username), "N/A")
    print(f"Leaderboard updated. {username}: {new_score*100:.2f}% — Rank #{rank_pos}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leaderboard/update.py score.json")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    username = os.environ.get("GITHUB_ACTOR", "unknown")
    update_leaderboard(data["accuracy"], username)
