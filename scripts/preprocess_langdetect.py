import pandas as pd
from langdetect import detect


# Load the clean data from the last step:
df = pd.read_csv('../data/processed/cleaned_data_example.csv')

# detect language in "all text" column and add language abbreviation to new column:
df['language'] = df['all text'].apply(lambda text: detect(text) if isinstance(text, str) and text.strip() != '' else 'unknown')

# Save the updated DataFrame to a new CSV file
df.to_csv('../data/processed/langdetect_data_example.csv', index=False)

print("Language detection complete.")