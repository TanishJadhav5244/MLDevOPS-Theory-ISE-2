import numpy as np
import pandas as pd
from typing import Tuple
from src.config import DATA_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

def generate_fintech_dataset(
    n_samples: int = 5000,
    drift_type: str = "none",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic FinTech Credit Default Risk Data.
    Drift modes:
      - 'none': In-distribution baseline reference data (Stable market).
      - 'covariate_shift': Feature Drift. Higher inflation/interest rate, higher DTI, higher revolving utilization.
      - 'concept_drift': Structural change in P(Y|X). Economic crisis where even mid-high credit score
                         borrowers with business loans default at higher rates.
    """
    rng = np.random.default_rng(random_state)

    # 1. Base Feature Distributions
    if drift_type == "covariate_shift":
        # Data Drift / Feature Drift: Macroeconomic pressure
        applicant_age = rng.integers(21, 65, size=n_samples)
        annual_income = rng.lognormal(mean=10.6, sigma=0.55, size=n_samples) # Slightly depressed income
        credit_score = rng.normal(loc=640, scale=60, size=n_samples)       # Shifted left (worse score)
        debt_to_income = rng.normal(loc=0.45, scale=0.12, size=n_samples)   # Higher DTI
        loan_amount = rng.normal(loc=22000, scale=8000, size=n_samples)     # Borrowing more
        interest_rate = rng.normal(loc=17.5, scale=3.5, size=n_samples)     # Substantial rate hike (11% -> 17.5%)
        revolving_utilization = rng.beta(a=4.5, b=2.5, size=n_samples)      # High credit card utilization
        open_credit_lines = rng.integers(3, 20, size=n_samples)
        employment_length = rng.integers(0, 15, size=n_samples)
        delinquencies_2yrs = rng.poisson(lam=0.6, size=n_samples)           # More recent delinquencies
        
        home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n_samples, p=[0.55, 0.35, 0.10])
        loan_purpose = rng.choice(
            ["debt_consolidation", "credit_card", "small_business", "home_improvement", "education"],
            size=n_samples,
            p=[0.45, 0.30, 0.15, 0.05, 0.05]
        )
        employment_type = rng.choice(["salaried", "self_employed", "contract"], size=n_samples, p=[0.60, 0.25, 0.15])

    elif drift_type == "concept_drift":
        # Feature distributions are moderately shifted, but the RELATIONSHIP to Default changes drastically!
        applicant_age = rng.integers(22, 65, size=n_samples)
        annual_income = rng.lognormal(mean=10.9, sigma=0.5, size=n_samples)
        credit_score = rng.normal(loc=670, scale=50, size=n_samples)
        debt_to_income = rng.normal(loc=0.38, scale=0.10, size=n_samples)
        loan_amount = rng.normal(loc=18000, scale=7000, size=n_samples)
        interest_rate = rng.normal(loc=13.0, scale=2.5, size=n_samples)
        revolving_utilization = rng.beta(a=3.0, b=3.5, size=n_samples)
        open_credit_lines = rng.integers(2, 18, size=n_samples)
        employment_length = rng.integers(1, 20, size=n_samples)
        delinquencies_2yrs = rng.poisson(lam=0.3, size=n_samples)
        
        home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n_samples, p=[0.45, 0.45, 0.10])
        loan_purpose = rng.choice(
            ["debt_consolidation", "credit_card", "small_business", "home_improvement", "education"],
            size=n_samples,
            p=[0.30, 0.25, 0.30, 0.10, 0.05]  # Surge in small business loans
        )
        employment_type = rng.choice(["salaried", "self_employed", "contract"], size=n_samples, p=[0.50, 0.35, 0.15])

    else:
        # Baseline In-Distribution
        applicant_age = rng.integers(22, 65, size=n_samples)
        annual_income = rng.lognormal(mean=11.0, sigma=0.5, size=n_samples) # Median ~$60,000
        credit_score = rng.normal(loc=690, scale=50, size=n_samples)       # Fair-to-Good
        debt_to_income = rng.normal(loc=0.32, scale=0.08, size=n_samples)
        loan_amount = rng.normal(loc=15000, scale=6000, size=n_samples)
        interest_rate = rng.normal(loc=11.5, scale=2.5, size=n_samples)
        revolving_utilization = rng.beta(a=2.5, b=4.0, size=n_samples)
        open_credit_lines = rng.integers(2, 16, size=n_samples)
        employment_length = rng.integers(1, 22, size=n_samples)
        delinquencies_2yrs = rng.poisson(lam=0.2, size=n_samples)
        
        home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n_samples, p=[0.40, 0.50, 0.10])
        loan_purpose = rng.choice(
            ["debt_consolidation", "credit_card", "small_business", "home_improvement", "education"],
            size=n_samples,
            p=[0.35, 0.30, 0.15, 0.12, 0.08]
        )
        employment_type = rng.choice(["salaried", "self_employed", "contract"], size=n_samples, p=[0.70, 0.20, 0.10])

    # Clean clip bounds
    credit_score = np.clip(credit_score, 300, 850)
    debt_to_income = np.clip(debt_to_income, 0.05, 0.85)
    loan_amount = np.clip(loan_amount, 1000, 50000)
    interest_rate = np.clip(interest_rate, 4.0, 32.0)
    revolving_utilization = np.clip(revolving_utilization, 0.01, 0.99)

    # 2. Ground-truth Default Generation Logic
    if drift_type == "concept_drift":
        # CONCEPT DRIFT:
        # Relationship P(Y|X) changes! Small businesses & self-employed default rapidly,
        # traditional credit scores lose predictive power (weights inversion).
        logit = (
            - 1.5
            - 0.002 * (credit_score - 670)          # Credit score weaker defense
            + 3.5 * debt_to_income
            + 0.15 * interest_rate
            + 2.5 * revolving_utilization
            + 0.8 * delinquencies_2yrs
            + np.where(loan_purpose == "small_business", 2.2, 0.0)      # High vulnerability shock
            + np.where(employment_type == "self_employed", 1.8, 0.0)
            + np.where(home_ownership == "RENT", 0.6, 0.0)
        )
    else:
        # Standard Credit Risk formula
        logit = (
            - 3.2
            - 0.009 * (credit_score - 680)
            + 4.0 * debt_to_income
            + 0.08 * interest_rate
            + 2.0 * revolving_utilization
            + 0.5 * delinquencies_2yrs
            - 0.04 * employment_length
            + np.where(loan_purpose == "small_business", 0.7, 0.0)
            + np.where(home_ownership == "RENT", 0.4, 0.0)
        )

    prob = 1.0 / (1.0 + np.exp(-logit))
    loan_default = (rng.random(size=n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "applicant_age": applicant_age.astype(int),
        "annual_income": np.round(annual_income, 2),
        "credit_score": np.round(credit_score, 1),
        "debt_to_income_ratio": np.round(debt_to_income, 4),
        "loan_amount": np.round(loan_amount, 2),
        "interest_rate": np.round(interest_rate, 2),
        "employment_length_years": employment_length.astype(int),
        "open_credit_lines": open_credit_lines.astype(int),
        "revolving_utilization_rate": np.round(revolving_utilization, 4),
        "delinquencies_2yrs": delinquencies_2yrs.astype(int),
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "employment_type": employment_type,
        TARGET_COL: loan_default
    })

    return df

def generate_and_save_all_datasets():
    print("[-] Generating Baseline Reference Data...")
    df_ref = generate_fintech_dataset(n_samples=10000, drift_type="none", random_state=101)
    df_ref.to_csv(DATA_DIR / "baseline_reference.csv", index=False)

    print("[-] Generating Production Batch 1 (In-Distribution / No Drift)...")
    df_prod1 = generate_fintech_dataset(n_samples=2500, drift_type="none", random_state=202)
    df_prod1.to_csv(DATA_DIR / "production_batch1_no_drift.csv", index=False)

    print("[-] Generating Production Batch 2 (Covariate Shift / Data Drift)...")
    df_prod2 = generate_fintech_dataset(n_samples=2500, drift_type="covariate_shift", random_state=303)
    df_prod2.to_csv(DATA_DIR / "production_batch2_data_drift.csv", index=False)

    print("[-] Generating Production Batch 3 (Concept Drift / Structural Shift)...")
    df_prod3 = generate_fintech_dataset(n_samples=2500, drift_type="concept_drift", random_state=404)
    df_prod3.to_csv(DATA_DIR / "production_batch3_concept_drift.csv", index=False)

    print(f"[+] All datasets generated successfully and saved to {DATA_DIR}!")

if __name__ == "__main__":
    generate_and_save_all_datasets()
