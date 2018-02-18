"""
Created on Wed Feb 14 16:35:54 2018

@author: andrewbates
"""

import numpy as np
from scipy import stats


'''
find_beta computes the hyperparameters for a Beta(a,b) prior given the mode and a percentile
'''

def find_beta(mode, percentile, prob = 0.95, b_start = 1, b_end = 100, b_length = 1000):

    b = np.linspace(b_start, b_end, num = b_length)
    a = ( 1 + mode * (b - 2) ) / (1 - mode)
    q  = stats.beta.ppf(prob, a, b)
    indx = np.argmax(q < percentile)
    return np.array([a[indx], b[indx]])

'''
Example usage
Suppose you're modeling your data as Binomial(n,p) with p ~ Beta(a,b).
Your expert says the most likely value for p is 0.1 and they are 95% certain p is no larger than 0.25.
To find the parameters a,b to capture this information, use
find_beta(mode = 0.1, percentile = 0.25)
'''
