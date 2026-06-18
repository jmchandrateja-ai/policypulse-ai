import os
import json
import pandas as pd
from groq import Groq
import time

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def generate(domain, sentiment, stance, urgency, count=15):
    prompt = f"""Generate {count} realistic Indian citizen policy complaints.
Domain: {domain}
Sentiment: {sentiment}
Stance: {stance}
Urgency: {urgency}
Rules:
- Write like a real Indian citizen
- Use simple English with Indian context
- 1-2 sentences each
- Return ONLY a JSON array of strings, nothing else"""

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

# Wait for rate limit to reset
print("Waiting 10 seconds for rate limit reset...")
time.sleep(10)

df = pd.read_csv("data/policy_complaints.csv")
print(f"Current samples: {len(df)}")

new_rows = []
for sentiment, stance in [("negative", "oppose"), ("positive", "support")]:
    print(f"Generating housing | {sentiment} | {stance}")
    complaints = generate("housing", sentiment, stance, "medium")
    for c in complaints:
        new_rows.append({
            "text": c,
            "sentiment": 0 if sentiment == "negative" else 2,
            "stance": 0 if stance == "oppose" else 2,
            "domain": 4,
            "urgency": 1,
            "domain_name": "housing",
            "sentiment_name": sentiment,
            "stance_name": stance,
            "urgency_name": "medium",
        })
    print(f"  Generated {len(complaints)} samples")
    time.sleep(5)

new_df = pd.DataFrame(new_rows)
df = pd.concat([df, new_df], ignore_index=True)
df.to_csv("data/policy_complaints.csv", index=False)
print(f"\nUpdated dataset: {len(df)} total samples")
print(df["domain_name"].value_counts())
