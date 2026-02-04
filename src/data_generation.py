import numpy as np
import pandas as pd
import os


def generate_data(n=10000, seed=42):
    """Generate synthetic taxpayer data for compliance analysis."""
    np.random.seed(seed)

    df = pd.DataFrame({
        "taxpayer_id": range(n),
        "declared_income": np.random.gamma(4, 12000, n),
        "property_count": np.random.poisson(1.2, n),
        "director_flag": np.random.binomial(1, 0.15, n),
        "late_filing_count": np.random.poisson(0.6, n),
        "previous_penalty": np.random.binomial(1, 0.1, n)
    })

    risk_score = (
        0.00003 * df["declared_income"]
        + 0.6 * df["property_count"]
        + 1.2 * df["late_filing_count"]
        + 1.5 * df["previous_penalty"]
        + 1.0 * df["director_flag"]
    )

    df["risk_flag"] = (risk_score + np.random.normal(0, 1, n) > 3.5).astype(int)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/synthetic_taxpayers.csv", index=False)
    return df
