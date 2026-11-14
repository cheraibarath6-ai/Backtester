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

def test_stats_reports_net_winner_and_loser_bps():
    x=stats(np.array([100.,-50.,25.]),np.array([1,-1,1],dtype=np.int8),np.array([2,1,2],dtype=np.int8))
    assert x['net_winning_bps']==125.
    assert x['net_losing_bps']==50.
    assert x['net_bps']==75.


def test_official_funding_quality_accepts_complete_eight_hour_series():
    from crypto_research.model3_top2k_transfer import funding_quality
    ix=pd.date_range('2024-01-01',periods=4,freq='8h',tz='UTC')
    f=pd.DataFrame({'funding_rate':[.0001]*4,'funding_interval_hours':[8]*4},index=ix)
    f.attrs['source']='binance_vision_official_monthly_fundingRate'
    q=funding_quality(f,'BTCUSDT')
    assert q['source'].startswith('binance_vision_official')
    assert q['suspicious_gap_count']==0
    assert q['coverage_pct']==100.


def test_official_funding_quality_flags_missing_settlement():
    from crypto_research.model3_top2k_transfer import funding_quality
    ix=pd.DatetimeIndex([pd.Timestamp('2024-01-01',tz='UTC'),
                         pd.Timestamp('2024-01-02',tz='UTC'),
                         pd.Timestamp('2024-01-03',tz='UTC')])
    f=pd.DataFrame({'funding_rate':[.0001]*3,'funding_interval_hours':[8]*3},index=ix)
    f.attrs['source']='binance_vision_official_monthly_fundingRate'
    q=funding_quality(f,'BTCUSDT')
    assert q['suspicious_gap_count']==1
    assert q['estimated_missing_events']==2
    assert q['coverage_pct']<100.
