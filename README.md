# Close Rate / Probability of Bind Model

This project builds a **classification model for close rate**, with the goal of estimating the **probability that a quote will bind** from a quote into a policy.

The intent is to serve as a portfolio piece: if you arrived here from my personal website, this repo is a clear, well-structured example of how I approach training and evaluating classification models focused on **probability quality**, not just raw discrimination.

```

## Problem Overview

In a typical insurance workflow, many quotes are issued but only a fraction bind. Understanding **which quotes are more likely to bind** is useful for:

- Prioritizing underwriting and marketing effort
- Evaluating pricing and competitiveness strategies
- Monitoring agent or segment-level performance

**Goal:**
Given quote-level features (e.g. account characteristics, agent, zip code, territory, etc.), estimate the probability of bind.

This is a **binary classification** problem where the target is:

- `1` → quote bound (policy issued)  
- `0` → quote did not bind (quote closed / lost / expired)

## Data

This is all the insurance quotes that have come into the company from March 2019 to December 2020 in the state of Indiana. It includes characteristics about the potential policyholders.

In a more typical production environment, a few things would be different:

- I'd rely on far more data to build a model like this. The model's quality may not be that high given the size of this dataset.
- Data would be gathered by querying the company's quote database with a regular refresh cadence of some kind rather than relying on this static dataset.

## Repository Layout

This is the intended structure as the project develops:

```text
hit_rate_model/
├─ README.md
├─ .gitignore
├─ requirements.txt   # Python dependencies
├─ data/
│  ├─ raw/            # Raw data
│  └─ processed/      # Final modeling datasets
├─ notebooks/
└─ src/