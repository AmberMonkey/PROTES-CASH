# binary_switch_ml_experiment.py
# One-file framework with small/mid/big datasets, 20 binary hyperparameters,
# and a results table per dataset.

import argparse
import random
from typing import Dict, Any, Tuple
from time import perf_counter as tpc
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, Normalizer,
    PolynomialFeatures
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------
# 20 binary switches (0 = off, 1 = on)
# -----------------------------
SWITCH_TEMPLATE: Dict[str, int] = {
    # Preprocessing (exclusive scaler; priority Standard > Robust > MinMax)
    "use_standard_scaler": 1,
    "use_robust_scaler": 0,
    "use_minmax_scaler": 0,

    # Transforms (can stack sequentially)
    "use_power_transformer": 0,   # Yeo-Johnson
    "use_normalize_l2": 0,        # Normalizer(norm='l2')

    # Feature engineering
    "use_polynomial_features": 0,  # degree=2
    "use_interactions_only": 0,    # only valid if polynomial features on
    "use_pca": 0,                  # placeholder (not used to keep deps light)
    "use_select_k_best": 0,        # ANOVA F-test selection
    "use_impute_median": 0,

    # Model family (priority: HGB > RF > LR)
    "use_logistic_regression": 1,
    "use_random_forest": 0,
    "use_hist_gradient_boosting": 0,

    # Model options
    "use_class_weight_balanced": 0,  # LR & RF
    "use_early_stopping": 0,         # HGB only
    "use_bootstrap": 0,              # RF only
    "use_max_depth_limit": 0,        # RF/HGB: if on, max_depth=6
    "use_warm_start": 0,             # RF only
    "use_l2_penalty": 1,             # LR: if off -> penalty='none'
    "use_calibration": 0,            # CalibratedClassifierCV(cv=3, isotonic)
}


# -----------------------------
# Datasets
# -----------------------------
def load_small(random_state: int = 42):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

def load_mid(random_state: int = 42, n_samples: int = 10000, n_features: int = 30):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(0, n_features // 5),
        n_repeated=0, n_classes=2,
        class_sep=1.2, flip_y=0.005, random_state=random_state,
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")

def load_big(random_state: int = 42, n_samples: int = 30000, n_features: int = 50):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(0, n_features // 4),
        n_repeated=0, n_classes=2,
        class_sep=1.1, flip_y=0.01, weights=[0.5, 0.5],
        random_state=random_state,
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")

def load_dataset(kind: str, random_state: int = 42):
    kind = kind.lower()
    if kind in ("small", "small_real", "breast_cancer"):
        return load_small(random_state)
    elif kind in ("mid", "medium"):
        return load_mid(random_state)
    elif kind in ("big", "large"):
        return load_big(random_state)
    raise ValueError(f"Unknown dataset kind '{kind}'. Choose: small | mid | big.")

def make_splits(X, y, random_state: int = 42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# -----------------------------
# Labels
# -----------------------------
def model_label(config: Dict[str, int]) -> str:
    if config.get("use_hist_gradient_boosting", 0):
        return "HGB"
    if config.get("use_random_forest", 0):
        return "RF"
    return "LR"

def scaler_label(config: Dict[str, int]) -> str:
    if config.get("use_standard_scaler", 0):
        return "Standard"
    if config.get("use_robust_scaler", 0):
        return "Robust"
    if config.get("use_minmax_scaler", 0):
        return "MinMax"
    return "None"


# -----------------------------
# Pipeline Builder (sequential preprocessing)
# -----------------------------
def build_pipeline(config: Dict[str, int],
                   n_samples: int,
                   n_features: int,
                   allow_heavy_models: bool,
                   random_state: int = 42):
    # Sequential numeric preprocessing pipeline
    numeric_steps = []
    if config.get("use_impute_median", 0):
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    if config.get("use_standard_scaler", 0):
        numeric_steps.append(("scaler", StandardScaler()))
    elif config.get("use_robust_scaler", 0):
        numeric_steps.append(("scaler", RobustScaler()))
    elif config.get("use_minmax_scaler", 0):
        numeric_steps.append(("scaler", MinMaxScaler()))
    if config.get("use_power_transformer", 0):
        numeric_steps.append(("power", PowerTransformer(method="yeo-johnson")))
    if config.get("use_normalize_l2", 0):
        numeric_steps.append(("norm", Normalizer(norm="l2")))

    steps = []
    if numeric_steps:
        numeric_pipe = SkPipeline(numeric_steps)
        steps.append(("pre", ColumnTransformer([("num", numeric_pipe, slice(0, None))],
                                               remainder="passthrough")))

    # Guard polynomial features via rough memory estimate
    allow_poly = bool(config.get("use_polynomial_features", 0))
    est_poly_out = n_features + (n_features * (n_features - 1)) // 2  # degree=2
    if allow_poly and (n_samples * est_poly_out > 6_000_000):
        allow_poly = False
    if allow_poly:
        steps.append(("poly", PolynomialFeatures(
            degree=2, include_bias=False,
            interaction_only=bool(config.get("use_interactions_only", 0))
        )))

    # Feature selection
    if config.get("use_select_k_best", 0):
        k = min(30, max(10, n_features // 2))
        steps.append(("select", SelectKBest(score_func=f_classif, k=k)))

    # Model (priority: HGB > RF > LR)
    if allow_heavy_models and config.get("use_hist_gradient_boosting", 0):
        model = HistGradientBoostingClassifier(
            early_stopping=bool(config.get("use_early_stopping", 0)),
            random_state=random_state,
            max_depth=6 if config.get("use_max_depth_limit", 0) else None,
            learning_rate=0.1,
            max_iter=50,  # fast default
        )
    elif allow_heavy_models and config.get("use_random_forest", 0):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6 if config.get("use_max_depth_limit", 0) else None,
            bootstrap=bool(config.get("use_bootstrap", 0)),
            warm_start=bool(config.get("use_warm_start", 0)),
            class_weight="balanced" if config.get("use_class_weight_balanced", 0) else None,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        penalty = "l2" if config.get("use_l2_penalty", 1) else None
        model = LogisticRegression(
            penalty=penalty, solver="saga", C=1.0, max_iter=600,
            class_weight="balanced" if config.get("use_class_weight_balanced", 0) else None,
            random_state=random_state, n_jobs=-1,
        )

    # Calibration (skip if heavy/off)
    if config.get("use_calibration", 0) and allow_heavy_models and (n_samples <= 40000):
        model = CalibratedClassifierCV(base_estimator=model, method="isotonic", cv=3)

    steps.append(("clf", model))
    return SkPipeline(steps)


# -----------------------------
# Evaluation (API for your optimizer)
# -----------------------------
def evaluate_config(config: Dict[str, int], data, allow_heavy_models: bool, random_state: int = 42):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    n_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    n_features = X_train.shape[1]

    pipe = build_pipeline(config, n_samples, n_features, allow_heavy_models, random_state)
    pipe.fit(X_train, y_train)

    def _proba(est, X):
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1]
        if hasattr(est, "decision_function"):
            s = est.decision_function(X)
            s = (s - s.min()) / (s.max() - s.min() + 1e-9)
            return s
        return est.predict(X)

    val_pred = _proba(pipe, X_val)
    test_pred = _proba(pipe, X_test)

    return {
        "val_auc": float(roc_auc_score(y_val, val_pred)),
        "test_auc": float(roc_auc_score(y_test, test_pred)),
        "config": dict(config),
    }


# -----------------------------
# Random config (replace with your optimizer)
# -----------------------------
def random_binary_config(seed: int = 0, allow_heavy_models: bool = True):
    rng = random.Random(seed)
    cfg = {}
    for k in SWITCH_TEMPLATE:
        # Bias away from very heavy toggles for speed when needed
        if k in ("use_polynomial_features", "use_calibration"):
            cfg[k] = 0 if rng.random() < 0.7 else 1
        else:
            cfg[k] = rng.randint(0, 1)
    # Force at least one model on, respect allow_heavy_models
    if allow_heavy_models:
        if sum(cfg[k] for k in ["use_hist_gradient_boosting", "use_random_forest", "use_logistic_regression"]) == 0:
            cfg["use_logistic_regression"] = 1
    else:
        cfg["use_hist_gradient_boosting"] = 0
        cfg["use_random_forest"] = 0
        cfg["use_logistic_regression"] = 1
        cfg["use_calibration"] = 0
    
    return cfg


# -----------------------------
# Run search and save table
# -----------------------------
from protes import protes
def protes_solution(objective_function):
  d = 20
  n = 2
  m = 1000

  def black_box_function(x):
      results=[]
      for a in x:
          results.append(objective_function(a))
      return results

  fopt, iopt = protes(black_box_function, d, n,m,with_info_full=True)
  fopt=np.array(fopt,dtype=np.float32)
  
  return iopt,fopt

from mads import mads
def mads_solution(objective_function):
  design_variables = np.random.randint(0,2,20)                # Initial design variables
  bounds_lower = [0]*20                    # Lower bounds for design variables
  bounds_upper = [1]*20       # Upper bounds for design variables
  
  def mads_ob(P):
      R=[]
      for x in P:
          if(x<0.5):
              R.append(0)
          else:
              R.append(1)
      return objective_function(R)
  dp_tol = 1E-6                               # Minimum poll size stopping criteria
  nitermax = 1000                             # Maximum objective function evaluations
  dp = 0.05                                    # Initial poll size as percent of bounds
  dm = 0.05                                   # Initial mesh size as percent of bounds
  # Run the optimizer
  ans,parameters=mads.orthomads(design_variables, bounds_upper, bounds_lower, mads_ob, dp, dm, dp_tol, nitermax, False, False)
  return ans,parameters


def run_search_for_dataset(dataset_kind: str, n_trials: int, seed: int, allow_heavy_models: bool, out_csv: str):
    X, y = load_dataset(dataset_kind)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = make_splits(X, y)
    data = (X_train, y_train), (X_val, y_val), (X_test, y_test)

    results = []
    rng = random.Random(seed)
    def objective_function(P):
      t=tpc()
      D=SWITCH_TEMPLATE.copy()
      for i,x in enumerate(D):
        D[x]=P[i]
      res=evaluate_config(D,data=data,allow_heavy_models=allow_heavy_models)
      t=tpc()-t
      print(t,'\r',end='')
      return -res['val_auc']
    
    V,P=protes_solution(objective_function)
    D=SWITCH_TEMPLATE.copy()
    for i,x in enumerate(D):
      D[x]=P[i]
    res=evaluate_config(D,data=data,allow_heavy_models=allow_heavy_models)
    row = {
        "trial": 0,
        "val_auc": res["val_auc"],
        "test_auc": res["test_auc"],
        "model": model_label(D),
        "scaler": scaler_label(D),
    }
    row.update(res["config"])
    results.append(row)
    for t in range(1, n_trials + 1):
        continue#remove this to make random data
        cfg = random_binary_config(seed=rng.randint(0, 10_000_000), allow_heavy_models=allow_heavy_models)
        res = evaluate_config(cfg, data=data, allow_heavy_models=allow_heavy_models)
        row = {
            "trial": t,
            "val_auc": res["val_auc"],
            "test_auc": res["test_auc"],
            "model": model_label(cfg),
            "scaler": scaler_label(cfg),
        }
        row.update(res["config"])
        results.append(row)

    df = pd.DataFrame(results).sort_values("val_auc", ascending=False).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[{dataset_kind.upper()}] Top rows:")
    print(df[["trial", "model", "scaler", "val_auc", "test_auc"]].head(5).to_string(index=False))
    print(f"[{dataset_kind.upper()}] Saved table -> {out_csv}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", help="small | mid | big | all")
    ap.add_argument("--trials_small", type=int, default=8)
    ap.add_argument("--trials_mid",   type=int, default=6)
    ap.add_argument("--trials_big",   type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fast-demo", type=int, default=1, help="1=only LR + fewer heavy toggles; 0=allow RF/HGB/Calibration")
    args = ap.parse_args()

    kinds = ["small", "mid", "big"] if args.dataset == "all" else [args.dataset]
    trials_map = {"small": args.trials_small, "mid": args.trials_mid, "big": args.trials_big}
    allow_heavy_models = (args.fast_demo == 0)

    for k in kinds:
        out_csv = f"results_{k}.csv"
        run_search_for_dataset(
            dataset_kind=k,
            n_trials=trials_map[k],
            seed=args.seed + hash(k) % 1000,
            allow_heavy_models=allow_heavy_models,
            out_csv=out_csv,
        )


if __name__ == "__main__":
    main()
