def build_features(df):
    """Build feature matrix and target variable from dataframe."""
    features = [
        "declared_income",
        "property_count",
        "director_flag",
        "late_filing_count",
        "previous_penalty"
    ]

    X = df[features]
    y = df["risk_flag"]
    return X, y
