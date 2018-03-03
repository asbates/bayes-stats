"""

@author: andrewbates
"""

import numpy as np
import pandas as pd
from scipy import stats
import pymc3 as pm


'''
find_beta() computes the hyperparameters for a Beta(a,b) prior
 given the mode and a percentile
'''

def find_beta(mode, percentile, p = 0.95, b_start = 1, b_end = 100, b_length = 1000):

    b = np.linspace(b_start, b_end, num = b_length)
    a = ( 1 + mode * (b - 2) ) / (1 - mode)
    q  = stats.beta.ppf(p, a, b)
    indx = np.argmax(q < percentile)
    out =  pd.DataFrame({'a': [a[indx]] , 'b': [b[indx]] })
    return out

'''
note: this is another way using the index argument which is basically rownames in R  
out =  pd.DataFrame(data = [ a[indx], b[indx] ], index = ['a', 'b'])
    out = out.transpose()
    return out    
'''


'''
Example usage:
Suppose you're modeling your data as Binomial(n,p) with p ~ Beta(a,b).
Your expert says the most likely value for p is 0.1 and they are 95% certain p is no larger than 0.25.
To find the parameters a,b to capture this information, use
find_beta(mode = 0.1, percentile = 0.25)
'''



'''
find_normal() computes the hyperparameters for a normal prior
 given the prior mean and a percentile
'''

def find_normal(mean, percentile, p = 0.95):
    
    sd =  (percentile - mean) / stats.norm.ppf(p) 
    precision = 1 / (sd ** 2)
    params = pd.DataFrame({'mean': [mean], 'sd': [sd], 'precision': [precision]})
    return params
    



'''
bayes_summary() computes summary statistics and and HPD interval 
 given a posterior sample and a level for the interval
'''

def bayes_summary(sample, p = 0.95):
    
    df = pd.DataFrame(sample)
    stat_sum = df.describe()['mean' : 'max']
    hpd = pm.stats.hpd(sample, alpha = 1 - p)
    hpd_df = pd.DataFrame([hpd[0], hpd[1]])
    summary = stat_sum.append(hpd_df)
    summary = summary.transpose()
    cols = summary.columns.tolist()
    cols = cols[2:5] + cols[0:1] + cols[5:7] + cols[7:9]
    summary = summary[cols]
    summary.columns = ['min', 'Q1', 'median', 'mean', 'Q3', 'max', 'hpd_lwr', 'hpd_upr']
    return summary
    

'''
Example usage:
post_sample = stats.norm.rvs(size = 100)
bayes_summary(post_sample)
'''
























