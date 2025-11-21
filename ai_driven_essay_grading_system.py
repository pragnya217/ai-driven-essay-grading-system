import re
import spacy
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize stopwords
stp = set(stopwords.words('english'))
stp.add("would")

# ============================================================================
# DOCUMENT DATA
# ============================================================================

# Data directory path
DATA_DIR = "/data"

# Document definitions with their respective scores
DOCUMENTS = [
    ("d1_12.txt", 12),
    ("d2_12.txt", 12),
    ("d3_12.txt", 12),
    ("d4_10.txt", 10),
    ("d5_10.txt", 10),
    ("d6_10.txt", 10),
    ("d7_8.txt", 8),
    ("d8_8.txt", 8),
    ("d9_8.txt", 8),
    ("d10_6.txt", 6),
    ("d11_6.txt", 6),
    ("d12_6.txt", 6),
    ("d13_4.txt", 4),
    ("d14_4.txt", 4),
    ("d15_4.txt", 4),
    ("d16_2.txt", 2),
    ("d17_2.txt", 2),
    ("d18_2.txt", 2)
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def remove_punctuation_conv_lowercase_lemm(word_list):
    """Remove punctuation, convert to lowercase, and remove @ mentions"""
    clean_list = []
    punctuation_chars = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for word in word_list:
        clean_word = re.sub(r'\S*@\S*', '', word)
        clean_word = clean_word.lower().translate(str.maketrans('', '', punctuation_chars))
        clean_list.append(clean_word)
    return clean_list


def lemmatize_word(word):
    """Lemmatize a single word using SpaCy"""
    doc = nlp(word)
    return doc[0].lemma_


def process_document(filename):
    """
    Process a single document through the entire pipeline:
    1. Read file
    2. Clean and remove stopwords
    3. Lemmatize
    4. Calculate frequencies
    
    Returns: (sorted_lemmatized_words, term_frequencies, unique_terms)
    """
    # Construct full file path
    import os
    filepath = os.path.join(DATA_DIR, filename)
    
    # Read the file
    with open(filepath, "r") as f:
        text = f.read()
    
    # Split into words
    word_list = text.split()
    print(f"Processing {filename}: {len(word_list)} words before processing")
    
    # Clean the list
    sorted_list = sorted(word_list)
    cleaned_list = remove_punctuation_conv_lowercase_lemm(sorted_list)
    cleaned_list = [word for word in cleaned_list if word and word not in stp]
    
    # Lemmatize
    lemmatized_words = [lemmatize_word(word) for word in cleaned_list]
    lemmatized_words = [word for word in lemmatized_words if word and word not in stp]
    
    # Get unique terms and sort
    unique_terms = set(lemmatized_words)
    sorted_lemmatized_words = sorted(lemmatized_words)
    
    # Calculate frequencies
    term_frequencies = [[term, sorted_lemmatized_words.count(term)] for term in unique_terms]
    
    print(f"After processing {filename}: {len(sorted_lemmatized_words)} words, {len(unique_terms)} unique terms")
    
    return sorted_lemmatized_words, term_frequencies, list(unique_terms)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    # Step 1: Process all documents from /data directory
    print("Processing documents from /data directory...")
    all_sorted_lemmatized = []
    all_term_frequencies = []
    all_unique_terms = []
    
    for filename, score in DOCUMENTS:
        sorted_lem, term_freq, unique_terms = process_document(filename)
        all_sorted_lemmatized.append(sorted_lem)
        all_term_frequencies.append(term_freq)
        all_unique_terms.append(unique_terms)
    
    # Step 2: Find common elements (intersection of first 3 documents - grade 12)
    print("\n" + "="*70)
    print("Computing common elements for grade 12 essays...")
    common_elements = list(set(all_unique_terms[0]) & set(all_unique_terms[1]) & set(all_unique_terms[2]))
    print(f"Found {len(common_elements)} common terms")
    
    # Build common elements frequency table
    common_element_data = []
    for term in common_elements:
        row = [term] + [doc_lemmatized.count(term) for doc_lemmatized in all_sorted_lemmatized]
        common_element_data.append(row)
    
    # Sort by total frequency
    sorted_common_element = sorted(common_element_data, key=lambda x: sum(x[1:]), reverse=True)
    
    # Step 3: Find union of all elements
    print("\n" + "="*70)
    print("Computing union of all elements...")
    union_elements = set()
    for terms in all_unique_terms:
        union_elements.update(terms)
    union_elements = list(union_elements)
    print(f"Found {len(union_elements)} total unique terms across all essays")
    
    # Build union frequency table
    union_data = []
    for term in union_elements:
        row = [term] + [doc_lemmatized.count(term) for doc_lemmatized in all_sorted_lemmatized]
        union_data.append(row)
    
    # Sort by total frequency
    sorted_union_list = sorted(union_data, key=lambda x: sum(x[1:]), reverse=True)
    
    # Step 4: Create Excel files
    print("\n" + "="*70)
    print("Creating Excel output files...")
    
    # Common elements Excel
    df_common = pd.DataFrame(sorted_common_element, columns=['Term'] + [f'd{i}' for i in range(1, 19)])
    df_common = df_common.set_index('Term').T.reset_index()
    df_common = df_common.rename(columns={'index': 'Document'})
    df_common['Score'] = [score for _, score in DOCUMENTS]
    df_common.to_excel('output_common.xlsx', index=False)
    print("Created output_common.xlsx")
    
    # Union elements Excel
    df_union = pd.DataFrame(sorted_union_list, columns=['Term'] + [f'd{i}' for i in range(1, 19)])
    df_union = df_union.set_index('Term').T.reset_index()
    df_union = df_union.rename(columns={'index': 'Document'})
    df_union['Score'] = [score for _, score in DOCUMENTS]
    df_union.to_excel('output_union.xlsx', index=False)
    print("Created output_union.xlsx")
    
    # Step 5: Machine Learning Pipeline
    print("\n" + "="*70)
    print("Running Machine Learning Pipeline...")
    
    # Prepare dataframe for ML - read from /data directory
    import os
    essays = []
    scores = []
    for filename, score in DOCUMENTS:
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r") as f:
            essays.append(f.read())
        scores.append(score)
    
    df_full = pd.DataFrame({'essay': essays, 'score': scores})
    
    # Custom Feature Extractor
    class FeatureExtractor(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()
    
        def clean_and_tokenize(self, text):
            text = re.sub(r'[^\w\s]', '', text.lower())
            words = text.split()
            words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
            return words
    
        def transform(self, X):
            features = []
            for essay in X:
                words = essay.split()
                word_count = len(words)
                errors = sum(1 for w in words if not w.isalpha())
                error_rate = errors / word_count if word_count > 0 else 0
                sentences = re.split(r'[.!?]', essay)
                sentences = [s for s in sentences if s.strip()]
                avg_sentence_len = word_count / len(sentences) if sentences else 0
    
                clean_words = self.clean_and_tokenize(essay)
                word_freq = pd.Series(clean_words).value_counts()
                top_word_count = word_freq.iloc[0] if not word_freq.empty else 0
    
                features.append([word_count, error_rate, avg_sentence_len, top_word_count])
            return np.array(features)
    
        def fit(self, X, y=None):
            return self
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        df_full["essay"], df_full["score"], test_size=0.3, random_state=42
    )
    
    pipeline = Pipeline([
        ("features", FeatureExtractor()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    kappa = cohen_kappa_score(y_test, y_pred, weights="quadratic")
    
    # Output predictions
    results = pd.DataFrame({
        "Essay": X_test.values,
        "True Score": y_test.values,
        "Predicted Score": y_pred
    })
    
    print("\nMachine Learning Results:")
    print(results.to_string(index=False))
    print(f"\nQuadratic Weighted Kappa Score: {kappa:.2f}")
    print("\n" + "="*70)
    print("Processing complete!")


if __name__ == "__main__":
    main()
