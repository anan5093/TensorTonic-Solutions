import numpy as np
def perplexity(prob_distributions, actual_tokens):

    prob_distributions = np.array(prob_distributions)
    actual_tokens = np.array(actual_tokens)

    row_idx = np.arange(len(actual_tokens))
    target_prob = prob_distributions[row_idx, actual_tokens]

    cross_entropy = -np.mean(np.log(target_prob + 1e-15))

    return float(np.exp(cross_entropy))


    
