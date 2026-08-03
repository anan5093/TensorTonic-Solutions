import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    # Tokenize and build vocabulary
    tokenized_docs = [doc.lower().split() for doc in documents]
    vocab = sorted(list(set(word for doc in tokenized_docs for word in doc)))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    
    num_docs = len(documents)
    num_vocab = len(vocab)
    
    # Calculate Document Frequency (DF) for each word
    df_counter = Counter()
    for doc_tokens in tokenized_docs:
        df_counter.update(set(doc_tokens))
        
    # Initialize TF-IDF matrix
    tfidf_matrix = np.zeros((num_docs, num_vocab))
    
    # Fill the matrix with TF-IDF values
    for doc_idx, doc_tokens in enumerate(tokenized_docs):
        tf_counter = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        
        for word, count in tf_counter.items():
            if word in word_to_index:
                col_idx = word_to_index[word]
                # Term Frequency
                tf = count / doc_len if doc_len > 0 else 0
                # Inverse Document Frequency
                idf = math.log(num_docs / (df_counter[word]))
                # TF-IDF
                tfidf_matrix[doc_idx, col_idx] = tf * idf
                
    return tfidf_matrix, vocab
