import pandas as pd
from transformers import pipeline

# Load the CSV produced from the language detection step
df = pd.read_csv('../data/processed/langdetect_data_example.csv')
print("read the output from the previous step")

# import sentencepiece
# print("sentencepiece version:", sentencepiece.__version__)

# Get unique language codes, excluding English and 'unknown'
unique_langs = [lang for lang in df['language'].unique() if lang not in ['en', 'unknown']]
print("Detected non-English languages:", unique_langs)

def get_helsinki_translation_pipeline(lang):

    # Use language code to get appropriate Helsinki model
    model_name = f"Helsinki-NLP/opus-mt-{lang}-en"
    try:
        # The translation pipeline will automatically download the model if necessary.
        translator = pipeline("translation", model=model_name)
        print(f"Successfully loaded translation model for '{lang}' using '{model_name}'.")
        return translator
    except Exception as e:
        print(f"Failed to load translation model for '{lang}' using '{model_name}': {e}")
        print(f"No official model found for language '{lang}'. Consider looking for an unofficial package.")
        return None

# Create a dictionary mapping each non-English language to its translation pipeline
translation_pipelines = {}
for lang in unique_langs:
    translator = get_helsinki_translation_pipeline(lang)
    if translator is not None:
        translation_pipelines[lang] = translator

print("\nTranslation pipelines available for languages:")
for lang, pipe in translation_pipelines.items():
    print(f"  {lang}: {pipe.model.config.name_or_path}")


def translate_text(text, lang):
    if lang in translation_pipelines:
        try:
            result = translation_pipelines[lang](text)
            return result[0]['translation_text']
        except Exception as e:
            print(f"Error translating text for language '{lang}': {e}")
            return text
    else:
        return text

# Create a new column 'translated_text'
# For rows where language is not English, translate the 'all text' field.
df['translated_text'] = df.apply(
    lambda row: translate_text(row['all text'], row['language']) if row['language'] != 'en' else row['all text'],
    axis=1
)

# Save the updated DataFrame to a new CSV file
df.to_csv('../data/processed/translated_data_example.csv', index=False)
print("translation complete.")
