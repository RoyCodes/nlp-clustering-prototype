import pandas as pd
from transformers import pipeline

# Load the CSV produced from the language detection step
df = pd.read_csv('../data/processed/langdetect_data_example.csv')
print("read the output from the previous step")

# Get unique language codes, excluding English and 'unknown'
unique_langs = [lang for lang in df['language'].unique() if lang not in ['en', 'unknown']]
print("Detected non-English languages:", unique_langs)

def get_helsinki_translation_pipeline(lang):
    """
    Construct the model name for Helsinki-NLP for a given language code and load
    the translation pipeline to translate from that language to English.
    """
    # The convention for Helsinki models: e.g., for Korean ('ko'): "Helsinki-NLP/opus-mt-ko-en"
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
