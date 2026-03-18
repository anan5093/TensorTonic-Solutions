import numpy as np

def bag_of_words_vector(tokens, vocab):
    vocab_index = {word: i for i, word in enumerate(vocab)}

    counts = np.zeros(len(vocab), dtype = int)

    for token in tokens:
        if token in vocab_index:
            counts[vocab_index[token]] +=1
    return counts       
    
   # pass