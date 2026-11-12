from __future__ import annotations
import time, zipfile
from pathlib import Path
import requests
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/monthly"
COLS = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]

def months(start, end):
    return pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="MS", inclusive="left")

def get_zip(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100:
        return dest
    for attempt in range(4):
        try:
            r=requests.get(url,timeout=90)
            if r.status_code==404: return None
            r.raise_for_status()
            if not r.content.startswith(b"PK"): raise ValueError("Archive response is not ZIP")
            dest.write_bytes(r.content)
            return dest
        except Exception:
            if attempt==3: raise
            time.sleep(2*(attempt+1))

def read_zip(path: Path):
    with zipfile.ZipFile(path) as z:
        csvs=[n for n in z.namelist() if n.endswith(".csv")]
        if not csvs: raise ValueError(f"No CSV in {path}")
        with z.open(csvs[0]) as f: d=pd.read_csv(f,header=None)
    d=d.iloc[:,:12]; d.columns=COLS
    # Binance archive CSVs may include a header row; discard non-numeric timestamps safely.
    open_ms=pd.to_numeric(d["open_time"],errors="coerce")
    d=d.loc[open_ms.notna()].copy()
    d["open_time"]=pd.to_datetime(open_ms.loc[d.index],unit="ms",utc=True)
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d["trades"]=pd.to_numeric(d.trades,errors="coerce").fillna(0)
    d=d.dropna(subset=["open","high","low","close","volume"])
    return d.sort_values("open_time").drop_duplicates("open_time").set_index("open_time")

def acquire_klines(symbol,start,end,root=Path("data")):
    parts=[]
    for m in months(start,end):
        ym=m.strftime("%Y-%m"); name=f"{symbol}-1m-{ym}.zip"
        url=f"{BASE}/klines/{symbol}/1m/{name}"
        p=get_zip(url,root/"raw"/"klines"/symbol/name)
        if p: parts.append(read_zip(p))
    if not parts: raise RuntimeError(f"No monthly kline archives found for {symbol}")
    d=pd.concat(parts).sort_index()
    d=d[~d.index.duplicated(keep="last")]
    return d.loc[(d.index>=pd.Timestamp(start,tz="UTC"))&(d.index<pd.Timestamp(end,tz="UTC"))]

def read_funding_zip(path: Path):
    with zipfile.ZipFile(path) as z:
        csvs=[n for n in z.namelist() if n.endswith(".csv")]
        if not csvs: raise ValueError(f"No funding CSV in {path}")
        with z.open(csvs[0]) as f: d=pd.read_csv(f)
    d.columns=[str(x).strip().lower() for x in d.columns]
    time_col=next((x for x in ["calc_time","funding_time","fundingtime"] if x in d.columns),None)
    rate_col=next((x for x in ["last_funding_rate","funding_rate","fundingrate"] if x in d.columns),None)
    if time_col is None or rate_col is None:
        raise ValueError(f"Unexpected Binance funding schema in {path}: {list(d.columns)}")
    d["funding_time_ms"]=pd.to_numeric(d[time_col],errors="coerce")
    d["funding_rate"]=pd.to_numeric(d[rate_col],errors="coerce")
    if "funding_interval_hours" in d.columns:
        d["funding_interval_hours"]=pd.to_numeric(d["funding_interval_hours"],errors="coerce")
    else:
        d["funding_interval_hours"]=8
    d=d.dropna(subset=["funding_time_ms","funding_rate"])
    d["funding_time"]=pd.to_datetime(d.funding_time_ms,unit="ms",utc=True)
    return d.drop_duplicates("funding_time").set_index("funding_time")[["funding_rate","funding_interval_hours"]].sort_index()

def acquire_funding(symbol,start,end,root=Path("data")):
    # Use Binance Vision's own monthly USD-M fundingRate archives. No third-party fallback:
    # if official archive coverage is incomplete, fail visibly rather than silently mixing provenance.
    parts=[]
    for m in months(start,end):
        ym=m.strftime("%Y-%m"); name=f"{symbol}-fundingRate-{ym}.zip"
        url=f"{BASE}/fundingRate/{symbol}/{name}"
        p=get_zip(url,root/"raw"/"fundingRate"/symbol/name)
        if p: parts.append(read_funding_zip(p))
    if not parts:
        raise RuntimeError(f"No official Binance Vision funding archives found for {symbol}")
    out=pd.concat(parts).sort_index()
    out=out[~out.index.duplicated(keep="last")]
    out=out.loc[(out.index>=pd.Timestamp(start,tz="UTC"))&(out.index<pd.Timestamp(end,tz="UTC"))]
    if out.empty:
        raise RuntimeError(f"Official Binance Vision funding data is empty for {symbol}")
    out.attrs["source"]="binance_vision_official_monthly_fundingRate"
    return out

def quality_report(d,symbol):
    expected=pd.date_range(d.index.min(),d.index.max(),freq="1min",tz="UTC")
    bad=((d.high<d[["open","close","low"]].max(axis=1))|(d.low>d[["open","close","high"]].min(axis=1))|(d.high<d.low)).sum()
    return {"symbol":symbol,"rows":len(d),"start":str(d.index.min()),"end":str(d.index.max()),"missing_minutes":len(expected.difference(d.index)),"duplicate_timestamps":int(d.index.duplicated().sum()),"bad_ohlc_rows":int(bad)}
