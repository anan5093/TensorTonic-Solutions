def word_count_dict(sentences):
    
    freq = {}

    for sentence in sentences:
        for word in sentence:
            freq[word] = freq.get(word, 0) +1
    return freq        
    
    
    # Your code here
    #pass