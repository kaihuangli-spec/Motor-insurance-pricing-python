"""Motor insurance pure premium modelling with Poisson and Gamma GLMs.

Data sources: OpenML freMTPL2freq (41214) and freMTPL2sev (41215).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["DrivAge", "VehAge", "BonusMalus"]
CATEGORICAL_FEATURES = ["VehGas", "Area"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42
MAX_POLICIES = 100_000
RESULTS = Path("results")


def load_data():
    """Combine policy exposures and claim amounts using the policy identifier."""
    frequency_data = fetch_openml(data_id=41214, as_frame=True).data.copy()
    severity_data = fetch_openml(data_id=41215, as_frame=True).data.copy()

    frequency_data["IDpol"] = pd.to_numeric(frequency_data["IDpol"]).astype("int64")
    severity_data["IDpol"] = pd.to_numeric(severity_data["IDpol"]).astype("int64")
    severity_data["ClaimAmount"] = pd.to_numeric(
        severity_data["ClaimAmount"], errors="coerce"
    )

    # Aggregate multiple claim payments at the policy level.
    total_amounts = (
        severity_data.groupby("IDpol")["ClaimAmount"]
        .sum(min_count=1)
        .rename("ClaimAmount")
    )
    data = frequency_data.join(total_amounts, on="IDpol")
    data["ClaimAmount"] = data["ClaimAmount"].fillna(0)
    original_count = len(data)

    for column in NUMERIC_FEATURES + ["ClaimNb", "Exposure", "ClaimAmount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        data[column] = (
            data[column].astype("string").str.strip("'").fillna("Unknown")
        )

    data = data.dropna(
        subset=NUMERIC_FEATURES + ["ClaimNb", "Exposure", "ClaimAmount"]
    )
    data = data[
        (data["Exposure"] > 0)
        & (data["ClaimNb"] >= 0)
        & (data["ClaimAmount"] >= 0)
    ].copy()

    # Limit selected extreme values and enforce consistent claim targets.
    data["Exposure"] = data["Exposure"].clip(upper=1)
    data["ClaimNb"] = data["ClaimNb"].clip(upper=4)
    data["ClaimAmount"] = data["ClaimAmount"].clip(upper=200_000)
    data = data[(data["ClaimNb"] > 0) == (data["ClaimAmount"] > 0)].copy()
    cleaned_count = len(data)

    if cleaned_count < 200:
        raise ValueError("Insufficient valid policy records after cleaning.")

    # Fixed-seed sampling makes the analysis reproducible.
    data = data.sample(n=min(MAX_POLICIES, cleaned_count), random_state=RANDOM_STATE)
    data_quality = pd.DataFrame(
        {
            "measure": [
                "policies_before_cleaning",
                "policies_after_cleaning",
                "policies_excluded",
                "policies_in_sample",
            ],
            "value": [original_count, cleaned_count, original_count - cleaned_count, len(data)],
        }
    )
    return data, data_quality


def model_pipeline(estimator):
    """Apply the same preprocessing structure to both regression models."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def main():
    RESULTS.mkdir(exist_ok=True)
    data, data_quality = load_data()
    train, test = train_test_split(data, test_size=0.25, random_state=RANDOM_STATE)
    train, test = train.copy(), test.copy()
    data_quality.to_csv(RESULTS / "data_quality.csv", index=False)

    print(f"Policies in sample: {len(data):,}")
    print(f"Training policies: {len(train):,}; test policies: {len(test):,}")
    print("Observed training claim frequency:",
          round(train["ClaimNb"].sum() / train["Exposure"].sum(), 4))
    print("Observed training claim severity:",
          round(train["ClaimAmount"].sum() / train["ClaimNb"].sum(), 2))

    # Examine the distribution of claim amounts for policies with paid claims.
    positive_amounts = data.loc[data["ClaimAmount"] > 0, "ClaimAmount"]
    plt.figure(figsize=(9, 5))
    plt.hist(np.log1p(positive_amounts), bins=35, edgecolor="white")
    plt.title("Distribution of positive policy claim amounts")
    plt.xlabel("log(1 + claim amount)")
    plt.ylabel("Number of policies")
    plt.tight_layout()
    plt.savefig(RESULTS / "claim_amount_distribution.png", dpi=150)
    plt.close()

    # Frequency model: claims per exposure-year, weighted by policy exposure.
    frequency_model = model_pipeline(PoissonRegressor(alpha=0.001, max_iter=400))
    frequency_model.fit(
        train[FEATURES],
        train["ClaimNb"] / train["Exposure"],
        model__sample_weight=train["Exposure"],
    )

    # Severity model: positive average paid amount, weighted by claim count.
    positive_train = train[
        (train["ClaimNb"] > 0) & (train["ClaimAmount"] > 0)
    ].copy()
    if len(positive_train) < 30:
        raise ValueError("Insufficient positive claims to fit the severity model.")
    severity_model = model_pipeline(GammaRegressor(alpha=0.1, max_iter=400))
    severity_model.fit(
        positive_train[FEATURES],
        positive_train["ClaimAmount"] / positive_train["ClaimNb"],
        model__sample_weight=positive_train["ClaimNb"],
    )

    # Expected claim cost per exposure-year is frequency times severity.
    test["PredictedFrequency"] = frequency_model.predict(test[FEATURES])
    test["PredictedSeverity"] = severity_model.predict(test[FEATURES])
    test["PurePremium"] = test["PredictedFrequency"] * test["PredictedSeverity"]
    test["PredictedClaims"] = test["PredictedFrequency"] * test["Exposure"]
    test["ExpectedClaims"] = test["PurePremium"] * test["Exposure"]

    actual_claims = test["ClaimAmount"].sum()
    expected_claims = test["ExpectedClaims"].sum()
    ratio = expected_claims / actual_claims if actual_claims > 0 else np.nan
    total_exposure = test["Exposure"].sum()
    performance = pd.DataFrame(
        {
            "measure": [
                "training_policies",
                "test_policies",
                "test_exposure_years",
                "observed_test_claims_cost",
                "predicted_test_claims_cost",
                "predicted_to_observed_ratio",
                "observed_test_frequency",
                "predicted_test_frequency",
                "observed_test_pure_premium",
                "predicted_test_pure_premium",
            ],
            "value": [
                len(train),
                len(test),
                total_exposure,
                actual_claims,
                expected_claims,
                ratio,
                test["ClaimNb"].sum() / total_exposure,
                test["PredictedClaims"].sum() / total_exposure,
                actual_claims / total_exposure,
                expected_claims / total_exposure,
            ],
        }
    )
    performance.to_csv(RESULTS / "model_performance.csv", index=False)
    print(f"Observed test claims cost: {actual_claims:,.2f}")
    print(f"Predicted test claims cost: {expected_claims:,.2f}")
    print(f"Predicted / observed cost ratio: {ratio:.3f}")

    # Compare observed and predicted claim frequency across driver age groups.
    age_edges = [0, 25, 35, 50, 65, 120]
    age_labels = ["0-24", "25-34", "35-49", "50-64", "65+"]
    test["AgeBand"] = pd.cut(
        test["DrivAge"], bins=age_edges, labels=age_labels, right=False
    )
    by_age = test.groupby("AgeBand", observed=True).agg(
        exposure=("Exposure", "sum"),
        actual_claims=("ClaimNb", "sum"),
        predicted_claims=("PredictedClaims", "sum"),
    )
    by_age["ObservedFrequency"] = by_age["actual_claims"] / by_age["exposure"]
    by_age["PredictedFrequency"] = by_age["predicted_claims"] / by_age["exposure"]
    by_age.to_csv(RESULTS / "frequency_by_age.csv")

    by_age[["ObservedFrequency", "PredictedFrequency"]].plot(
        kind="bar",
        figsize=(9, 5),
        rot=0,
        title="Observed versus predicted claim frequency by driver age",
    )
    plt.ylabel("Claims per exposure-year")
    plt.xlabel("Driver age band")
    plt.tight_layout()
    plt.savefig(RESULTS / "frequency_by_driver_age.png", dpi=150)
    plt.close()

    test[
        [
            "DrivAge", "VehAge", "BonusMalus", "Exposure", "ClaimNb",
            "ClaimAmount", "PredictedFrequency", "PredictedSeverity", "PurePremium",
        ]
    ].head(1000).to_csv(RESULTS / "sample_predictions.csv", index=False)
    print("Saved model outputs to the results directory.")


if __name__ == "__main__":
    main()
