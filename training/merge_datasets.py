import pandas as pd

print("Loading datasets...")

# 1. LLM-generated dataset
df_llm = pd.read_csv("data/policy_complaints.csv")
print(f"LLM generated: {len(df_llm)} samples")

# 2. Railway complaints - fix encoding
df_rail = pd.read_csv("data/raw/train.csv", encoding="latin-1")
df_rail = df_rail.rename(columns={"SentimentText": "text", "Sentiment": "sentiment"})
df_rail = df_rail[["text", "sentiment"]].dropna()
df_rail["stance"]         = 1
df_rail["domain"]         = 6
df_rail["urgency"]        = 1
df_rail["domain_name"]    = "transportation"
df_rail["sentiment_name"] = df_rail["sentiment"].map({0: "negative", 1: "positive"})
df_rail["stance_name"]    = "neutral"
df_rail["urgency_name"]   = "medium"
df_rail = df_rail[df_rail["sentiment"].isin([0, 1])]
df_rail = df_rail.sample(min(100, len(df_rail)), random_state=42)
print(f"Railway complaints: {len(df_rail)} samples")

# 3. Combine
df_master = pd.concat([df_llm, df_rail], ignore_index=True)
df_master = df_master.dropna(subset=["text"])
df_master["text"] = df_master["text"].astype(str).str.strip()
df_master = df_master[df_master["text"].str.len() > 10]

df_master.to_csv("data/master_dataset.csv", index=False)

print(f"\nMaster dataset: {len(df_master)} total samples")
print(f"\nDomain distribution:")
print(df_master["domain_name"].value_counts())
print(f"\nSentiment distribution:")
print(df_master["sentiment"].value_counts())
