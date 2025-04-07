[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Conda-environment-green.svg)](https://docs.conda.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![pandas](https://img.shields.io/badge/pandas-%3E%3D1.0-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%3E%3D1.18-blue.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%3E%3D3.0-blue.svg)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%3E%3D0.24-green.svg)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-%3E%3D3.0-purple.svg)](https://spacy.io/)
[![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-latest-blueviolet.svg)](https://www.sbert.net/)
[![Transformers](https://img.shields.io/badge/Transformers-latest-orange.svg)](https://huggingface.co/transformers/)
[![langdetect](https://img.shields.io/badge/langdetect-latest-lightgrey.svg)](https://pypi.org/project/langdetect/)
[![Helsinki-NLP](https://img.shields.io/badge/Translation-Helsinki--NLP-red.svg)](https://github.com/Helsinki-NLP/Opus-MT)

# Case Clustering AI: Unsupervised NLP for Support Tickets

## Intro

This project sets up a Python environment with all of the tools needed to cluster and glean insights from datasets like case correspondence from customer support inquiries. The goal is to translate all cases into English and then cluster them into similar topics. From here, these clusters can be reviewed and enriched with meaningful resolution codes that can drive product improvements.

**Sample Output:**

| Cluster  | Support Case Snippet |
| ------------- | ------------- |
| 0  | “I can’t find the subscription page in the app…”  |
| 1  | “The mobile feature crashes on startup…”  |
| 2  | “How do I reset my password?” |

## Tools

* **Conda** for environment setup and reproducibility  
* **Jupyter** for interactive development and testing  
* **pandas** and **numpy** for data manipulation and analysis  
* **matplotlib** and **seaborn** for data visualization    
* **langdetect** for language detection  
* **Helsinki-NLP** for language translation
**spaCy** for tokenizing the translated text in preparation for clustering.
* **scikit-learn** for K-means clustering and other machine learning tasks

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
7. **Open `notebook.ipynb`. Run the first cell to confirm everything is working.**
8. **Continue through the notebook.**