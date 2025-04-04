import pandas as pd
import spacy

# Load the CSV produced from the Helsinki NLP step
df = pd.read_csv('../data/processed/translated_data_example.csv')
print("read the output from the previous step")

# Load spaCy model to verify
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded successfully!")

# Tokenize the translated text using spaCy
def tokenize_text(text):
    # Change empty cells into empty strings
    if not isinstance(text, str):
        text = ""
    # Load the text into spaCy
    doc = nlp(text)
    # Filter out punctuation and whitespace; lowercase the token text.
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    return " ".join(tokens)

# Apply tokenization to the 'translated_text' column
df['tokenized_text'] = df['translated_text'].apply(tokenize_text)

# Save the updated DataFrame with tokenized text to a new CSV file
df.to_csv('../data/processed/tokenized_data_example.csv', index=False)
print("Tokenization complete.")