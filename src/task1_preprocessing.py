import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

DATA_PATH = "data/raw/complaints.csv"

print("Loading sample data...")

df = pd.read_csv(
    DATA_PATH,
    skiprows=lambda x: x > 0 and x % 100 != 0,
    low_memory=False,
)

print("Dataset loaded successfully!")
print()

print("Shape: ")
print(df.shape)
print(df["Consumer complaint narrative"].notna().sum())

print()
print("Columns: ")
print(df.columns.tolist())

print(df.head())


print("\nProduct Distribution")
print("-" * 50)

product_counts = df["Product"].value_counts()

print(product_counts)


plt.figure(figsize = (12,6))

sns.barplot(
    x=product_counts.index,
    y=product_counts.values,
)

plt.xticks(rotation=90)
plt.title("Complaint Distribution by Product")
plt.xlabel("Product")
plt.ylabel("Number of Complaints")

plt.tight_layout()
plt.show()


print("\n Narrative Availability")
print("-" * 50)

with_narrative = df["Consumer complaint narrative"].notna().sum()
without_narrative =df["Consumer complaint narrative"].isna().sum()

print(f"With narrative: {with_narrative}")
print(f"Without narrative: {without_narrative}")

plt.figure(figsize=(6,4))

sns.barplot(
    x=["With Narrative", "Without Narrative"],
    y=[with_narrative, without_narrative],
)

plt.title("Narrative Availability")
plt.ylabel("Count")

plt.show()


df["Narrative Length"] = (
    df["Consumer complaint narrative"]
    .fillna("")
    .str.split()
    .str.len()
)

print("\nNarrative Length Statistics")
print("-" * 50)

print(df["Narrative Length"].describe())

very_short = (df["Narrative Length"] < 10).sum()

very_long = (df["Narrative Length"] > 1000).sum()

print(f"\nVery short narratives (<10 words): {very_short:,}")
print(f"\nVery long narratives (>1000 words): {very_long:,}")


plt.figure(figsize=(10,6))

sns.histplot(
    df["Narrative Length"],
    bins=50
)

plt.title("Distribution of Narrative Lengths")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()


target_products = [
    "Credit card",
    "Credit card or prepaid card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
    "Money transfers",
    "Payday loan, title loan, or personal loan",
    "Payday loan, title loan, personal loan, or advance loan",
    "Consumer Loan"
]

filtered_df = df[
    df["Product"].isin(target_products)
].copy()

print("\nAfter Product Filtering")
print("-" * 50)
print(filtered_df.shape)

print(filtered_df["Product"].value_counts())

#remiving missing values

filtered_df = filtered_df[
    filtered_df["Consumer complaint narrative"].notna()
].copy()

filtered_df = filtered_df[
    filtered_df["Consumer complaint narrative"]
    .str.strip() #removes spaces
    .ne("")
].copy()


print("\nAfter Narrative Filtering")
print("-" * 50)

print(filtered_df.shape)

print(
    filtered_df["Consumer complaint narrative"]
    .isna()
    .sum()
)

print("\nRemaining Products")
print("-" * 50)

print(filtered_df["Product"].value_counts())


def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()

    text = re.sub(
        r"i am writing to file a complaint.*?",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

filtered_df["cleaned_narrative"] = (
    filtered_df["Consumer complaint narrative"]
    .apply(clean_text)
)


#verifying cleaning
print("\nCleaning Example")
print("-" * 50)

print(filtered_df[[
    "Consumer complaint narrative",
    "cleaned_narrative"
]].head(3))

print(filtered_df.columns.tolist())

print("ORIGINAL:")
print(filtered_df["Consumer complaint narrative"].iloc[0])

print("\nCLEANED:")
print(filtered_df["cleaned_narrative"].iloc[0])


output_path = "data/processed/filtered_complaints.csv"

filtered_df.to_csv(
    output_path,
    index=False
)

print("\nSaved cleaned dataset:")
print(output_path)