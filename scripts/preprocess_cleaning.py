import pandas as pd

# GOAL 1: add columns to flag missing data
# GOAL 2: combine free text fields and remove blank space

# Begin processing input CSV file
df = pd.read_csv('../data/raw/raw_data_example.csv')

# Grabbing the 3 free-text columns from the input data
col2 = df.iloc[:, 1]
col3 = df.iloc[:, 2]
col4 = df.iloc[:, 3]

# Add columns for missing free text
df['text 1 missing'] = col2.apply(lambda x: 'Y' if pd.isna(x) or str(x).strip() == '' else 'N')
df['text 2 missing'] = col3.apply(lambda x: 'Y' if pd.isna(x) or str(x).strip() == '' else 'N')
df['text 3 missing'] = col4.apply(lambda x: 'Y' if pd.isna(x) or str(x).strip() == '' else 'N')

# strip blank space, then cluster free text fields to new column
df['all text'] = (
    col2.fillna('').str.strip() + ' ' +
    col3.fillna('').str.strip() + ' ' +
    col4.fillna('').str.strip()
).str.strip()

# Save the modified dataframe to a new CSV file
df.to_csv('../data/processed/cleaned_data_example.csv', index=False)

print("Raw data has been cleaned")