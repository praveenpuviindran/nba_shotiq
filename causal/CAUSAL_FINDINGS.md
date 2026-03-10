# Causal Analysis: Does Shot Selection Cause Scoring Efficiency?

## Research Question

**Does a player improving their shot selection quality — measured by the average
expected make probability (xFG%) of shots taken — cause a measurable improvement
in their actual points per shot, beyond what their pre-existing shooting skill
would predict?**

This question matters for coaching: if shot selection *causes* efficiency gains,
coaches are justified in enforcing shot discipline.  If the relationship is purely
correlational (better shooters naturally take higher-quality shots), the case for
intervention is weaker.

---

## Methods Used

### 1. Inverse Probability Weighting (IPW)

- **Treatment**: player is in the top tercile of shot selection quality for that
  season (average xFG% ≥ 66th percentile — taking shots the model considers
  easier to make on average).
- **Outcome**: actual points per shot.
- **Confounders** (included in propensity model):
  - Prior season actual FG%
  - Prior season average xFG%
  - Prior season shot volume
  - 3-point attempt rate (controls for shot-type strategy)
  - Average shot distance
- **Estimator**: Horvitz-Thompson IPW estimator with clipped propensity scores
  (0.05, 0.95) to limit extreme weights.
- **Inference**: 500-iteration bootstrap for standard error and 95% CI.

### 2. Difference-in-Differences (DiD)

- **Treatment group**: players who *improved* their avg xFG% by > 0.5 SD
  between consecutive seasons.
- **Control group**: players with stable avg xFG% (change < 0.5 SD).
- **Outcome**: actual points per shot.
- **DiD estimator**:
  `δ = (treated_post − treated_pre) − (control_post − control_pre)`
- **Key assumption**: parallel trends — treated and control groups would have
  followed the same trajectory in the absence of the treatment.

### 3. Rosenbaum Sensitivity Analysis

Tests how large an unmeasured confounder would need to be (in odds-ratio terms,
Γ) to explain away the observed effect.  Gamma is increased until the upper-bound
p-value exceeds 0.05.

---

## Results

*These results are populated by running the analysis against the predictions table.
Run `python causal/shot_selection_analysis.py` or use the Causal Analysis tab in
the Streamlit app to populate live results.*

| Analysis | Estimate | 95% CI | p-value |
|---|---|---|---|
| IPW — ATE (pts/shot) | — | — | — |
| DiD — δ (pts/shot) | — | — | — |
| Rosenbaum Γ threshold | — | — | — |

---

## Interpretation

A positive and statistically significant ATE from IPW would suggest that players
who take higher-quality shots (higher average xFG%) score more points per shot
than comparable players who do not — after accounting for prior shooting skill and
shot volume.

The DiD estimate adds a temporal dimension: if players who *actively change* their
shot selection improve their scoring efficiency more than players who maintain
stable habits, this supports a causal narrative.

The Rosenbaum Γ quantifies robustness: a Γ of 1.5 means the findings hold unless
an unmeasured confounder makes the treatment 50% more likely for the improving
group — a substantial hurdle.

---

## Limitations

- **Observational data**: players are not randomly assigned to shot-selection
  strategies.  Unmeasured factors (coaching, defensive attention, injury) could
  drive both shot selection and outcomes simultaneously.
- **Parallel trends assumption**: the DiD design assumes treated and control
  players would have followed the same efficiency trend without the improvement
  in shot selection.  This cannot be directly verified.
- **Endogeneity**: a player's shot selection quality is partly determined by their
  own skill — defenders sag off poor shooters, pushing them into worse shots.
  The propensity model controls for observed confounders but cannot fully
  address this.
- **Sample size**: with only the seasons available in the database, the analysis
  may be underpowered for players with short track records.

---

## Code

```python
import sys
sys.path.insert(0, '.')

from src.config import DB_PATH
from causal.shot_selection_analysis import (
    load_player_season_data,
    compute_propensity_scores,
    estimate_ipw_effect,
    difference_in_differences,
    sensitivity_analysis,
)

# Load player-season panel
panel = load_player_season_data(DB_PATH)

# IPW
df_with_ps = compute_propensity_scores(panel)
ipw_result = estimate_ipw_effect(df_with_ps)
print("IPW ATE:", ipw_result)

# DiD
did_result = difference_in_differences(panel)
print("DiD δ:", did_result["did_estimate"])

# Rosenbaum bounds
sens_result = sensitivity_analysis(df_with_ps)
print(sens_result["interpretation"])
```
