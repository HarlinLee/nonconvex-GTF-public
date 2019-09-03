import pandas as pd
import os
import fnmatch
import numpy as np
import csv
from datetime import datetime
from scipy.stats import ttest_rel

prefix = ''
suffix = '.csv'
delim = '-'
datadir = '../datasets/UCI_data/results/'
#datasets = ['iris', 'heart-disease', 'yeast', 'wine', 'winequality','breast-cancer','internet-ads', 'car']
penalties = ['SCAD-L1', 'MCP-L1']
ks = np.arange(2)
outfn = datadir + 'pvalues' + datetime.now().strftime('-%y-%m-%d-%H-%M')+'.csv'
num_trials = 10

with open(outfn, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(
        ['filename', 'k', 'penalty_f', 'avg miscls rate', 'tstat', 'pvalue'])

    for f in os.listdir(datadir):
        if 'pvalues' not in f and fnmatch.fnmatch(f, '*.csv'):  #rip regex
            print (f)
            results = pd.read_csv(datadir + f, skiprows=1, dtype={'penalty_f': str, 'avg miscls rate': np.float64},
                                  names=['k', 'penalty_f', 'gamma', 'rho', 'penalty_param', 'avg miscls rate',
                                         'num_trial']+list(range(num_trials)))
            print (results.shape)
            for k in ks:
                l1_results = results.loc[(results['k'] == k) & (results['penalty_f'] == 'L1'), range(num_trials)]
                writer.writerow(
                    [f, k, 'L1', np.around(l1_results.values[0].mean(), decimals=4), None, None])
                for penalty in penalties:
                    print (penalty)
                    compared_results = results.loc[(results['k'] == k) & (results['penalty_f'] == penalty), range(num_trials)]
                    tstat, pvalue = ttest_rel(l1_results.values[0], compared_results.values[0])
                    print (tstat, pvalue)
                    writer.writerow([f, k, penalty, np.around(compared_results.values[0].mean(), decimals=4), tstat, pvalue])
