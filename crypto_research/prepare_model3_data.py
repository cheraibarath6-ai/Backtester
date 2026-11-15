from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import acquire_klines, acquire_funding, quality_report
from .sweep_m3_full_retracement import SYMBOLS, START, END

def funding_quality(f):
    if f.empty:
        raise RuntimeError("Official Binance funding data is empty")
    intervals=pd.to_numeric(f.funding_interval_hours,errors="coerce").fillna(8.).to_numpy(float)
    gaps=f.index.to_series().diff().dropna()
    suspicious=[]
    for i,delta in enumerate(gaps.to_numpy()):
        hours=float(delta/np.timedelta64(1,"h"))
        expected=max(float(intervals[i]),float(intervals[i+1]))
        if hours>1.75*expected:
            suspicious.append({"from":str(f.index[i]),"to":str(f.index[i+1]),
                               "gap_hours":hours,"expected_interval_hours":expected})
    return {"rows":len(f),"start":str(f.index.min()),"end":str(f.index.max()),
            "source":f.attrs.get("source","unknown"),
            "median_interval_hours":float(np.median(intervals)),
            "max_gap_hours":float(gaps.max().total_seconds()/3600) if len(gaps) else 0.,
            "suspicious_gap_count":len(suspicious),"suspicious_gaps":suspicious[:20]}

def main():
    out=Path("artifacts/preflight"); out.mkdir(parents=True,exist_ok=True)
    report=[]
    for s in SYMBOLS:
        print(f"Preflight {s}: acquiring official klines and funding",flush=True)
        d=acquire_klines(s,START,END)
        f=acquire_funding(s,START,END)
        q=quality_report(d,s)
        fq=funding_quality(f)
        q["funding_quality"]=fq
        report.append(q)
        print(json.dumps(q),flush=True)
        if q["missing_minutes"]!=0 or q["duplicate_timestamps"]!=0 or q["bad_ohlc_rows"]!=0:
            raise RuntimeError(f"{s}: candle data quality failure: {q}")
        if not fq["source"].startswith("binance_vision_official_monthly_fundingRate") or fq["suspicious_gap_count"]>0:
            raise RuntimeError(f"{s}: official funding data quality failure: {fq}")
    (out/"data_quality.json").write_text(json.dumps(report,indent=2))
    print(f"PREPARED {len(SYMBOLS)} ASSETS; all candle/funding checks passed",flush=True)

if __name__=="__main__": main()
