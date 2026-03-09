
import json,os
from datetime import datetime

scores_file="scores.json"
board="leaderboard/README.md"

scores=json.load(open(scores_file)) if os.path.exists(scores_file) else []

data=json.load(open("score.json"))
user=os.environ.get("GITHUB_ACTOR","local")

scores.append({"user":user,"score":data["accuracy"],"date":datetime.utcnow().strftime("%Y-%m-%d")})

scores.sort(key=lambda x:x["score"],reverse=True)

json.dump(scores,open(scores_file,"w"),indent=2)

lines=[
"# Leaderboard",
"| Rank | User | Score | Date |",
"|------|------|------|------|"
]

for i,s in enumerate(scores[:20]):
    lines.append(f"| {i+1} | {s['user']} | {s['score']:.4f} | {s['date']} |")

os.makedirs("leaderboard",exist_ok=True)
open(board,"w").write("\n".join(lines))
