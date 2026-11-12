import unittest
import pandas as pd
import numpy as np
from crypto_research.sweep_m3_full_retracement import summarize

class SmokeTests(unittest.TestCase):
    def test_metrics_empty(self):
        cols=["net_bps","net_R"]
        result=summarize(pd.DataFrame(columns=cols))
        self.assertEqual(result["trades"],0)
    def test_metrics_nonempty(self):
        t=pd.DataFrame({"net_bps":[100.0,-50.0,100.0],"net_R":[1.0,-0.5,1.0]})
        result=summarize(t)
        self.assertEqual(result["trades"],3)
        self.assertAlmostEqual(result["win_rate_pct"],66.67,places=2)
        self.assertAlmostEqual(result["profit_factor"],4.0,places=3)

if __name__=="__main__":
    unittest.main()
