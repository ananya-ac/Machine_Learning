import pdb
import numpy as np
import pandas as pd

def giniIndex(counts):
   
    l,r=counts
    if not l.empty:
        gl=1-sum((l/sum(l))**2)
        dl=sum(l)/(sum(l)+sum(r))
        gl*=dl
    else: gl=0
    if not r.empty:
        gr=1-sum((r/sum(r))**2)
        dr=sum(r)/(sum(l)+sum(r))
        gr*=dr
    else: gr=0
    
    return gl+gr

def entropy(counts):
    return

