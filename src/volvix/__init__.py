"""volvix: portfolio volatility forecasting with LSTM/Transformer and the incremental value of VIX.

Implements the locked protocol in configs/protocol.yaml (see docs/PROTOCOL_TH.md).
"""

__version__ = "1.0.0"

TICKERS = ("ACWI", "SHY", "XLK", "XLV", "GLD")
