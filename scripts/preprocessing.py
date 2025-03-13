import pandas as pd
import json
import time
import uuid

# GOAL 1: add columns to flag missing data
# GOAL 2: combine free text fields and remove blank space

# Begin processing input CSV file
df = pd.read_csv('raw-data.csv')

# Add columns for missing free text
df['1 missing'] = None
df['2 missing'] = None
df['3 missing'] = None

# strip blank space, then cluster free text fields to new column
text_fields = df['1'].str.strip() + ' ' + df['2'].str.strip() + ' ' + df['3'].str.strip()

# Save the modified dataframe to a new CSV file
df.to_csv('pre-processed-data.csv', index=False)

df['AnyBlankFlag'] = df[['FirstName', 'LastName', 'Age']].apply(lambda x: 'Y' if any(x == '') else 'N', axis=1)