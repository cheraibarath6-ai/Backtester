import numpy as np
import pandas as pd
from crypto_research.model3_top2k_transfer import bt, stats


def test_numba_stop_first_same_bar():
    base=pd.Timestamp('2024-01-01',tz='UTC').value
    ts=base+np.arange(5,dtype=np.int64)*3_600_000_000_000
    o=np.array([100.,100.,105.,105.,105.]); h=np.array([101.,110.,108.,106.,106.]); l=np.array([99.,100.,99.,104.,104.]); c=np.array([100.,109.,105.,105.,105.])
    sig=np.array([0,1,0,0,0],dtype=np.int8); fts=np.empty(0,dtype=np.int64); fp=np.array([0.])
    net,sides,reasons=bt(ts,o,h,l,c,sig,fts,fp,base,base+5*3_600_000_000_000,.75,2.,3,24)
    assert len(net)==1 and int(reasons[0])==1


def test_summary_pf():
    x=stats(np.array([100.,-50.,25.]),np.array([1,-1,1],dtype=np.int8),np.array([2,1,2],dtype=np.int8))
    assert x['trades']==3 and x['profit_factor']==2.5 and x['long_trades']==2