import pandas as pd
import subprocess
import json
import os

hashtags = [
    "NEP2020",
    "BBMPnotice",
    "KarnatakaPolicy",
    "PropertyTax",
    "GovernmentHospital",
    "IndiaEducation",
    "SmartCity",
    "MyGov",
]

all_tweets = []

for tag in hashtags:
    print(f"Scraping #{tag}...")
    try:
        result = subprocess.run(
            ["snscrape", "--jsonl", "--max-results", "50",
             "twitter-hashtag", tag],
            capture_output=True, text=True, timeout=60
        )
        for line in result.stdout.strip().split("\n"):
            if line:
                try:
                    tweet = json.loads(line)
                    text = tweet.get("content", "")
                    if text and len(text) > 20:
                        all_tweets.append({
                            "text": text,
                            "source": f"#{tag}",
                            "sentiment": -1,
                            "stance": -1,
                            "domain": -1,
                            "urgency": -1,
                        })
                except:
                    continue
        print(f"  Collected {len(all_tweets)} total so far")
    except Exception as e:
        print(f"  Error: {e}")

df = pd.DataFrame(all_tweets)
os.makedirs("data", exist_ok=True)
df.to_csv("data/raw_tweets.csv", index=False)
print(f"\nTotal tweets: {len(df)}")
print("Saved to data/raw_tweets.csv")
