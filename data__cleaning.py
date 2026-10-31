import pandas as pd

df = pd.read_csv("data/og.csv")

print(df.head())
print("\nDataset shape:", df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nContract distribution:")
print(df["Contract"].value_counts())

print("\nChurn rate by contract (%):")
print(
    df.groupby("Contract")["Churn"]
    .apply(lambda x: x.eq("Yes").mean() * 100)
    .round(2)
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

print("\nMissing TotalCharges:", df["TotalCharges"].isnull().sum())

df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:", df.duplicated().sum())



print("\nCleaned dataset saved successfully.")
print("Final dataset shape:", df.shape)