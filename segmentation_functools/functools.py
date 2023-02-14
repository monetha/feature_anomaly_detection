import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

def applyParallel(dfGrouped, func, *kwards):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group,*kwards) for name, group in dfGrouped)
    return pd.concat(retLst)

def split(a, t):
    k, m = divmod(len(a), t)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(t))