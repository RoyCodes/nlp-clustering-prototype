import pandas as pd
import spacy

# Load the CSV produced from the Helsinki NLP step
df = pd.read_csv('../data/processed/translated_data_example.csv')
print("read the output from the previous step")

# Load spaCy model to verify
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded successfully!")

