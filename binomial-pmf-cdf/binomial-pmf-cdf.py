import math

def binomial_pmf_cdf(n, p, k):

    n = int(n)
    k = int(k)
    p = float(p)

    if k < 0 or k > n:
        return (0.0, 0.0)

    pmf = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    cdf = 0.0
    for i in range(k + 1):
        cdf += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    return (float(pmf), float(cdf))