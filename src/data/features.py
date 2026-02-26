from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_NUMERIC_FEATURES = ["loc_x", "loc_y", "distance", "angle", "is_3"]
MODEL_CATEGORICAL_FEATURES = ["shot_zone_basic", "shot_zone_area", "shot_zone_range"]


def _safe_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def infer_is_three_point(df: pd.DataFrame) -> pd.Series:
    """Infer whether each shot is a 3PA using multiple fallbacks."""
    shot_type = _safe_series(df, "shot_type")
    zone_basic = _safe_series(df, "shot_zone_basic")
    zone_range = _safe_series(df, "shot_zone_range")

    shot_distance = pd.to_numeric(df.get("shot_distance"), errors="coerce")
    loc_x = pd.to_numeric(df.get("loc_x"), errors="coerce")
    loc_y = pd.to_numeric(df.get("loc_y"), errors="coerce")

    by_type = shot_type.str.contains("3PT", case=False, regex=False)
    by_zone_basic = zone_basic.str.contains("3", case=False)
    by_zone_range = zone_range.str.contains(r"24\+", case=False, regex=True)

    corner_three = (shot_distance >= 22) & (loc_y <= 140) & (loc_x.abs() >= 220)
    above_break_three = shot_distance >= 23.75

    return (
        by_type | by_zone_basic | by_zone_range | corner_three.fillna(False) | above_break_three.fillna(False)
    ).astype(int)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add location-derived features used by shot-make models."""
    out = df.copy()
    out["loc_x"] = pd.to_numeric(out["loc_x"], errors="coerce")
    out["loc_y"] = pd.to_numeric(out["loc_y"], errors="coerce")

    out["distance"] = np.sqrt(np.square(out["loc_x"]) + np.square(out["loc_y"]))
    out["angle"] = np.arctan2(out["loc_y"], out["loc_x"])
    out["is_3"] = infer_is_three_point(out)

    for col in MODEL_CATEGORICAL_FEATURES:
        out[col] = _safe_series(out, col, default="Unknown")

    return out


def build_model_matrix(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    """
    Build model matrix with numeric and one-hot encoded zone features.

    If feature_columns is provided, output is aligned to that exact feature set.
    """
    features = add_engineered_features(df)

    numeric = features[MODEL_NUMERIC_FEATURES].copy()
    numeric = numeric.fillna(0.0)

    dummies = pd.get_dummies(
        features[MODEL_CATEGORICAL_FEATURES],
        prefix=MODEL_CATEGORICAL_FEATURES,
        dtype=int,
    )

    X = pd.concat([numeric, dummies], axis=1)

    if feature_columns is None:
        final_feature_columns = list(X.columns)
    else:
        final_feature_columns = feature_columns
        X = X.reindex(columns=final_feature_columns, fill_value=0)

    y: pd.Series | None = None
    if "shot_made_flag" in features.columns:
        y = pd.to_numeric(features["shot_made_flag"], errors="coerce").fillna(0).astype(int)

    return X, y, final_feature_columns
