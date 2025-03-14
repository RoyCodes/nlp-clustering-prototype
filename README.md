[![Python](https://img.shields.io/badge/Python-3.12.8-blue.svg)](https://www.python.org/downloads/)
[![Conda](https://img.shields.io/badge/Conda-Miniconda-green.svg)](https://docs.conda.io/en/latest/miniconda.html)
[![Jupyter](https://img.shields.io/badge/Jupyter-1.1.1-orange.svg)](https://jupyter.org)
[![pandas](https://img.shields.io/badge/pandas-2.2.3-blue.svg)](https://pandas.pydata.org)
[![numpy](https://img.shields.io/badge/numpy-2.2.2-blue.svg)](https://numpy.org)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.10.0-blue.svg)](https://matplotlib.org)
[![seaborn](https://img.shields.io/badge/seaborn-0.13.2-orange.svg)](https://seaborn.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-green.svg)](https://scikit-learn.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.8.4-blue.svg)](https://spacy.io)
[![langdetect](https://img.shields.io/badge/langdetect-1.0.9-lightgrey.svg)](https://pypi.org/project/langdetect/)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-3.4.1-blueviolet.svg)](https://www.sbert.net)

# Customer Case ML Project

## Intro

This project sets up a Python environment with all of the ML tools needed to analyze and glean insights from datasets like case correspondence from customer support inquiries. The goal is to translate all cases into English, cluster them into similar topics, enrich them with a meaningful resolution code, and create visualizations of the outcomes.

## Tools

* **Conda** for environment setup and reproducibility  
* **Jupyter** for interactive development and testing  
* **pandas** and **numpy** for data manipulation and analysis  
* **matplotlib** and **seaborn** for data visualization  
* **scikit-learn** for clustering and other machine learning tasks  
* **langdetect** for language detection  
* **spaCy** for natural language processing

## Installation

1. **Install Conda.**
2. **Clone this repo.**
3. **Create the environment by running:**  
   `conda env create --file=environment.yml`
4. **Activate your new environment:**
   `conda activate nlp-clustering-prototype`
5. **NOTE:** If using VS Code, you may need to restart it before the new environment is available
6. **Install an English language model for spaCy first by running:**
   `python -m spacy download en_core_web_sm`
6. **Run `test.ipynb` to confirm everything is working.**

## To-do

### Data Preprocessing

1. add columns to flag missing data
2. combine free text fields and remove blank space.

### NLP

1. add column for language id (langdetect)
2. translate to english (spaCy)
3. generate embeddings (spaCy)

### Clustering

1. Apply clustering algorithm (K-Means) to the embeddings (Scikit-learn)