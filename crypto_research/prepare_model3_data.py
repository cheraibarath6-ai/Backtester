from pathlib import Path
import json
from .data import acquire_klines,acquire_funding,quality_report
from .sweep_m3_full_retracement import SYMBOLS,START,END
def main():
    out=Path("artifacts/preflight"); out.mkdir(parents=True,exist_ok=True)
    report=[]
    for s in SYMBOLS:
        d=acquire_klines(s,START,END)
        f=acquire_funding(s,START,END)
        q=quality_report(d,s)
        q.update({"funding_rows":len(f),"funding_start":str(f.index.min()),"funding_end":str(f.index.max()),
                  "funding_source":f.attrs.get("source","binance_vision_official_monthly_fundingRate")})
        report.append(q)
        print(json.dumps(q),flush=True)
    (out/"data_quality.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__": main()
