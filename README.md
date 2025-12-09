This project analyzes U.S. investment-grade corporate credit spreads versus Treasuries
and builds a simple regime classification model (stress vs normal).

Data sources:
- ICE BofA US Corporate Index OAS (via FRED, series `BAMLC0A0CM`)
- U.S. 10Y Treasury yield (FRED, `DGS10`)
- Yield curve slope 10Y–2Y (FRED, `T10Y2Y`)

Tech stack:
- Python (conda env `spread_regime_2501`)
- pandas, numpy
- matplotlib / seaborn
- scikit-learn / statsmodels

