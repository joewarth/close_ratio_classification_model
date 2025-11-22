# utils.py

import numpy as np
import pandas as pd

def split_train_test_by_time(
    X: pd.DataFrame,
    y: pd.Series,
    yyyymm_col: str = "yyyymm",
    test_frac: float = 0.2,
):
    """
    Time-based train/test split using a yyyymm column in X.

    - Start from the most recent month.
    - Add months going backward until the test set
      contains at least `test_frac` of all rows.
    - Train = all earlier months, Test = these most recent months.

    Assumes X[yyyymm_col] looks like 202310 (int or string).
    """

    if not (0 < test_frac < 1):
        raise ValueError("test_frac must be between 0 and 1.")

    if yyyymm_col not in X.columns:
        raise ValueError(f"{yyyymm_col!r} not found in X.columns")

    X_local = X.copy()
    X_local[yyyymm_col] = X_local[yyyymm_col].astype(int)

    months = X_local[yyyymm_col]
    unique_months = np.sort(months.unique())
    n_total = len(X_local)

    if len(unique_months) < 2:
        raise ValueError("Need at least 2 distinct months to do a time-based split.")

    # Walk backward from the most recent month collecting months into test
    selected_test_months = []
    cum_test_n = 0

    for m in unique_months[::-1]:  # newest → oldest
        month_n = (months == m).sum()
        selected_test_months.append(m)
        cum_test_n += month_n

        frac = cum_test_n / n_total

        # Stop when we've reached or exceeded the requested fraction
        if frac >= test_frac:
            break

    # Safety: don't consume *all* months
    if len(selected_test_months) == len(unique_months):
        # keep at least one month for training
        selected_test_months = selected_test_months[:-1]
        cum_test_n = (months.isin(selected_test_months)).sum()
        frac = cum_test_n / n_total

    selected_test_months = np.array(selected_test_months)

    test_mask = months.isin(selected_test_months)
    X_test = X_local[test_mask]
    y_test = y.loc[X_test.index]

    X_train = X_local[~test_mask]
    y_train = y.loc[X_train.index]

    actual_frac = len(X_test) / n_total

    print(f"Train months: {np.setdiff1d(unique_months, selected_test_months)}")
    print(f"Test  months: {np.sort(selected_test_months)}")
    print(f"Requested test frac: {test_frac:.3f}, actual: {actual_frac:.3f}")

    return X_train, X_test, y_train, y_test