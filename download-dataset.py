from datasets import load_dataset

# Load the dataset
ds = load_dataset("ronnieaban/hs-code", split="train")

# Convert to pandas DataFrame
df = ds.to_pandas()

# Save as CSV
df.to_csv("hs_code_dataset.csv", index=False)