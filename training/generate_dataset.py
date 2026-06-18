import os
import json
import pandas as pd
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

DOMAINS = [
    "taxation", "healthcare", "infrastructure",
    "education", "housing", "environment", "transportation"
]

sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
stance_map    = {"oppose": 0, "neutral": 1, "support": 2}
urgency_map   = {"low": 0, "medium": 1, "high": 2}
domain_map    = {d: i for i, d in enumerate(DOMAINS)}

def generate_complaints(domain, sentiment, stance, urgency, count=15):
    prompt = f"""Generate {count} realistic Indian citizen policy complaints.
Domain: {domain}
Sentiment: {sentiment}
Stance: {stance}
Urgency: {urgency}
Rules:
- Write like a real Indian citizen
- Use simple English with Indian context
- 1-2 sentences each
- Return ONLY a JSON array of strings, nothing else
Example: ["complaint 1", "complaint 2"]"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=1000
    )

    text = response.choices[0].message.content.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)

combinations = [
    ("taxation",       "negative", "oppose",  "low"),
    ("taxation",       "positive", "support", "low"),
    ("taxation",       "neutral",  "neutral", "low"),
    ("healthcare",     "negative", "oppose",  "high"),
    ("healthcare",     "positive", "support", "low"),
    ("healthcare",     "neutral",  "neutral", "low"),
    ("infrastructure", "negative", "oppose",  "medium"),
    ("infrastructure", "positive", "support", "low"),
    ("education",      "negative", "oppose",  "medium"),
    ("education",      "positive", "support", "low"),
    ("housing",        "negative", "oppose",  "medium"),
    ("housing",        "positive", "support", "low"),
    ("environment",    "negative", "oppose",  "high"),
    ("environment",    "positive", "support", "low"),
    ("transportation", "negative", "oppose",  "medium"),
    ("transportation", "positive", "support", "low"),
]

all_data = []

print("Generating dataset using Groq LLM...")

for i, (domain, sentiment, stance, urgency) in enumerate(combinations):
    print(f"[{i+1}/{len(combinations)}] {domain} | {sentiment} | {stance} | {urgency}")
    try:
        complaints = generate_complaints(domain, sentiment, stance, urgency)
        for complaint in complaints:
            all_data.append({
                "text":           complaint,
                "sentiment":      sentiment_map[sentiment],
                "stance":         stance_map[stance],
                "domain":         domain_map[domain],
                "urgency":        urgency_map[urgency],
                "domain_name":    domain,
                "sentiment_name": sentiment,
                "stance_name":    stance,
                "urgency_name":   urgency,
            })
        print(f"  Generated {len(complaints)} samples")
    except Exception as e:
        print(f"  Error: {e}")

df = pd.DataFrame(all_data)
df.to_csv("data/policy_complaints.csv", index=False)
print(f"\nDataset saved: {len(df)} total samples")
print(df["domain_name"].value_counts())
