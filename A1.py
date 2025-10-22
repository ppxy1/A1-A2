import argparse, warnings, json, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             balanced_accuracy_score, confusion_matrix)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import matplotlib.pyplot as plt

if not hasattr(np, "bool"):
    np.bool = np.bool_

try:
    from tqdm.auto import tqdm
    tqdm_available = True
except Exception:
    tqdm_available = False
    def tqdm(x, **k): 
        return x

shap_available = True
try:
    import shap
except Exception:
    shap_available = False

xgb_available = True
try:
    import inspect
    import xgboost as xgb
    from xgboost import XGBClassifier
except Exception:
    xgb_available = False

def _fit_accepts(param: str) -> bool:
    try:
        return param in inspect.signature(XGBClassifier.fit).parameters
    except Exception:
        return False

if xgb_available:
    _HAS_CALLBACKS = _fit_accepts("callbacks")
    _HAS_ES        = _fit_accepts("early_stopping_rounds")
    _HAS_EVAL_SET  = _fit_accepts("eval_set")
    _HAS_VERBOSE   = _fit_accepts("verbose")

    from sklearn.model_selection import train_test_split as _split

    class SafeEarlyStopXGB(XGBClassifier):
        def __init__(self, eval_fraction=0.25, early_stopping_rounds=100, random_state=None, **kwargs):
            kwargs.setdefault("eval_metric", "aucpr")
            super().__init__(**kwargs)
            self.eval_fraction = float(eval_fraction)
            self.early_stopping_rounds = int(early_stopping_rounds)
            self._inner_rs = random_state

        def fit(self, X, y, **fit_params):
            if not self.early_stopping_rounds or self.early_stopping_rounds <= 0:
                return super().fit(X, y, **fit_params)
            Xtr, Xval, ytr, yval = _split(X, y, test_size=self.eval_fraction,
                                          stratify=y, random_state=self._inner_rs)
            kws = dict(fit_params)
            if _HAS_EVAL_SET:
                kws["eval_set"] = [(Xval, yval)]
            if _HAS_CALLBACKS:
                es = xgb.callback.EarlyStopping(rounds=self.early_stopping_rounds,
                                                save_best=True, metric_name="aucpr", maximize=True)
                kws["callbacks"] = [es]
            elif _HAS_ES:
                kws["early_stopping_rounds"] = self.early_stopping_rounds
            if _HAS_VERBOSE and "verbose" not in kws:
                kws["verbose"] = False
            return super().fit(Xtr, ytr, **kws)

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    y_col = next((c for c in df.columns if "death_event" in c.lower()), None)
    if y_col is None:
        raise ValueError(f"Target column 'death_event' not found. Columns: {list(df.columns)}")
    X = df.drop(columns=[y_col]).copy()
    y = df[y_col].astype(int)
    for c in X.columns:
        try: X[c] = pd.to_numeric(X[c], errors="raise")
        except Exception: pass
    return X, y, y_col

def build_preprocess(numeric_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols)
    ])

def tune_threshold_cv_fbeta(estimator, X_train, y_train, cv, beta=2.0, recall_floor=0.0,
                            label="Threshold CV (Fβ)", show_progress=True):
    prob_oof = np.zeros_like(y_train, dtype=float)
    splits = list(cv.split(X_train, y_train))
    iterator = tqdm(splits, desc=label, unit="fold") if (show_progress and tqdm_available) else splits
    for tr_idx, va_idx in iterator:
        est = clone(estimator)
        est.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X_train.iloc[va_idx])[:, 1]
        else:
            s = est.decision_function(X_train.iloc[va_idx])
            p = 1.0 / (1.0 + np.exp(-s))
        prob_oof[va_idx] = p

    precision, recall, thresholds = precision_recall_curve(y_train, prob_oof)
    b2 = beta**2
    fbeta = (1+b2) * precision[:-1] * recall[:-1] / (b2*precision[:-1] + recall[:-1] + 1e-12)

    idx = np.arange(len(thresholds))
    if recall_floor > 0:
        mask = recall[:-1] >= recall_floor
        if mask.any():
            idx = idx[mask]
            fbeta = fbeta[mask]

    best = int(idx[np.argmax(fbeta)])
    return float(thresholds[best])

def evaluate(fitted, name, Xte, yte, threshold=None):
    if hasattr(fitted, "predict_proba"):
        y_prob = fitted.predict_proba(Xte)[:, 1]
    elif hasattr(fitted, "decision_function"):
        s = fitted.decision_function(Xte); y_prob = 1.0/(1.0+np.exp(-s))
    else:
        y_pred = fitted.predict(Xte); y_prob = y_pred.astype(float)

    thr = 0.5 if threshold is None else threshold
    y_pred = (y_prob >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(yte, y_pred, labels=[0,1]).ravel()
    acc_0 = tn/(tn+fp) if (tn+fp) else np.nan
    acc_1 = tp/(tp+fn) if (tp+fn) else np.nan
    f1_0, f1_1 = f1_score(yte, y_pred, average=None, labels=[0,1])
    ap = average_precision_score(yte, y_prob)

    metrics = {
        "Model": name, "Thresh": thr,
        "Test_Accuracy": float(accuracy_score(yte, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(yte, y_pred)),
        "Test_F1": float(f1_score(yte, y_pred)),
        "F1_0": float(f1_0), "F1_1": float(f1_1),
        "Acc_0(Specificity)": float(acc_0), "Acc_1(Recall)": float(acc_1),
        "Test_AUC": float(roc_auc_score(yte, y_prob)),
        "Test_AUPRC": float(ap),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Pred_Pos": int(y_pred.sum()), "True_Pos": int(yte.sum()),
        "y_pred": y_pred,
    }
    return metrics, y_prob

def plot_roc(model_probs, yte, out_png):
    plt.figure()
    for label, y_prob in model_probs:
        fpr, tpr, _ = roc_curve(yte, y_prob)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Test)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close()

def shap_rf(pipe, X_ref, feature_names, out_prefix, max_n=200, show_progress=True):
    prep = pipe.named_steps["prep"]
    rf = pipe.named_steps["clf"]
    Xt = prep.transform(X_ref)
    if hasattr(Xt, "toarray"): Xt = Xt.toarray()
    Xt = np.asarray(Xt)
    n = min(max_n, Xt.shape[0])
    Xt_small = Xt[:n]

    try:
        if not shap_available:
            raise RuntimeError("shap not available")
        if show_progress and tqdm_available:
            print(f"[SHAP] RF computing on {n} samples ...")
        explainer = shap.TreeExplainer(rf)
        raw = explainer.shap_values(Xt_small)
        vals = raw
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        if hasattr(vals, "values"): vals = vals.values
        vals = np.asarray(vals)
        if vals.ndim == 3 and vals.shape[-1] == 2:
            vals = vals[..., 1]
        imp = np.mean(np.abs(vals), axis=0).reshape(-1)
        method = "SHAP (TreeExplainer)"
    except Exception:
        imp = np.asarray(rf.feature_importances_).reshape(-1)
        method = "Fallback: feature_importances_ (not SHAP)"

    feature_names = np.asarray(feature_names)
    order = np.argsort(-imp)

    plt.figure()
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, imp[order])
    plt.yticks(y_pos, feature_names[order])
    plt.gca().invert_yaxis()
    plt.xlabel("mean |SHAP value|" if "SHAP" in method else "mean importance")
    plt.title(f"RF Feature Importance — {method}")
    png = f"{out_prefix}_rf_shap_bar.png"
    plt.tight_layout(); plt.savefig(png, dpi=160, bbox_inches="tight"); plt.close()

    df_imp = pd.DataFrame({"feature": feature_names[order], "importance": imp[order]})
    csv = f"{out_prefix}_rf_shap.csv"
    df_imp.to_csv(csv, index=False)
    return png, csv, method

def shap_xgb_native(pipe, X_ref, feature_names, out_prefix, max_n=500, show_progress=True):
    if not xgb_available:
        return None, None
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["clf"]
    Xt = prep.transform(X_ref)
    if hasattr(Xt, "toarray"): Xt = Xt.toarray()
    Xt = np.asarray(Xt)
    n = min(max_n, Xt.shape[0])
    if show_progress and tqdm_available:
        print(f"[SHAP] XGB TreeSHAP computing on {n} samples ...")
    dtest = xgb.DMatrix(Xt[:n])
    contribs = model.get_booster().predict(dtest, pred_contribs=True)
    vals = contribs[:, :-1]
    mean_abs = np.abs(vals).mean(axis=0)

    feature_names = np.asarray(feature_names)
    order = np.argsort(-mean_abs)

    plt.figure()
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, mean_abs[order])
    plt.yticks(y_pos, feature_names[order])
    plt.gca().invert_yaxis()
    plt.xlabel("mean |contribution|")
    plt.title("XGB TreeSHAP (native)")
    png = f"{out_prefix}_xgb_shap_bar.png"
    plt.tight_layout(); plt.savefig(png, dpi=160, bbox_inches="tight"); plt.close()

    df_imp = pd.DataFrame({"feature": feature_names[order], "mean_abs_contrib": mean_abs[order]})
    csv = f"{out_prefix}_xgb_shap.csv"
    df_imp.to_csv(csv, index=False)
    return png, csv

def pick_best(rows_df: pd.DataFrame):
    ranked = rows_df.sort_values(
        ["Test_F1", "Test_AUPRC", "Balanced_Accuracy"], ascending=False
    ).reset_index(drop=True)
    best = ranked.iloc[0].to_dict()
    reason = ("Best model is chosen by highest Test F1-score; ties are broken by "
              "higher Test AUPRC, then by higher Balanced Accuracy.")
    return best, reason, ranked

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts_minority_first")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--recall_floor", type=float, default=0.0)
    ap.add_argument("--pos_weight_mult", type=float, default=1.0)
    ap.add_argument("--progress", type=int, default=1)
    args = ap.parse_args()

    SHOW_PROGRESS = (args.progress == 1 and tqdm_available)

    t0 = time.time()
    X, y, y_col = load_data(args.csv)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    features = list(X.columns)
    preprocess = build_preprocess(features)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    vc = ytr.value_counts()
    neg = int(vc.get(0, 0)); pos = int(vc.get(1, 0))
    print(f"[Train class counts] neg={neg}, pos={pos}, prevalence={pos/(neg+pos):.3f}")

    rows, model_probs, preds = [], [], {}

    print("[RF] training...")
    rf = RandomForestClassifier(
        random_state=args.seed, n_estimators=400, max_features="sqrt",
        n_jobs=-1, bootstrap=True, class_weight="balanced_subsample",
        max_depth=10, min_samples_split=5, min_samples_leaf=2
    )
    rf_pipe = Pipeline([("prep", preprocess), ("clf", rf)])
    rf_pipe.fit(Xtr, ytr)
    thr_rf = tune_threshold_cv_fbeta(
        rf_pipe, Xtr, ytr, cv, beta=args.beta, recall_floor=args.recall_floor,
        label="RF Threshold CV (Fβ)", show_progress=SHOW_PROGRESS
    )
    met_rf, prob_rf = evaluate(rf_pipe, "RandomForest (minority-first)", Xte, yte, threshold=thr_rf)
    preds["RF"] = met_rf.pop("y_pred")
    rows.append(met_rf); model_probs.append(("RandomForest", prob_rf))
    print(f"[RF] done. best thr={thr_rf:.4f}")

    if xgb_available:
        spw = float(neg / max(pos, 1)) * float(args.pos_weight_mult)
        print(f"[XGB] training... (scale_pos_weight={spw:.3f})")
        xgb_base = SafeEarlyStopXGB(
            objective="binary:logistic",
            n_estimators=1200, learning_rate=0.03, max_depth=4,
            subsample=0.9, colsample_bytree=0.9,
            min_child_weight=2.0, reg_lambda=1.0, gamma=0.0,
            tree_method="hist", n_jobs=-1, verbosity=0,
            scale_pos_weight=spw, max_delta_step=1,
            eval_fraction=0.25, early_stopping_rounds=100, random_state=args.seed
        )
        xgb_pipe = Pipeline([("prep", preprocess), ("clf", xgb_base)])
        xgb_pipe.fit(Xtr, ytr)
        thr_xgb = tune_threshold_cv_fbeta(
            xgb_pipe, Xtr, ytr, cv, beta=args.beta, recall_floor=args.recall_floor,
            label="XGB Threshold CV (Fβ)", show_progress=SHOW_PROGRESS
        )
        met_xgb, prob_xgb = evaluate(xgb_pipe, "XGBoost (minority-first)", Xte, yte, threshold=thr_xgb)
        preds["XGB"] = met_xgb.pop("y_pred")
        rows.append(met_xgb); model_probs.append(("XGBoost", prob_xgb))
        print(f"[XGB] done. best thr={thr_xgb:.4f}")
    else:
        print("[XGB] xgboost not available. Skipped.")

    res = pd.DataFrame(rows)
    best, why, ranked = pick_best(res)
    res_path = out/"two_models_metrics.csv"
    ranked.to_csv(res_path, index=False)

    print("Two Models")
    print(ranked.to_string(index=False))
    print(f"\nBest model: {best['Model']}")
    print(f"Test_F1={best['Test_F1']:.4f}, Test_AUPRC={best['Test_AUPRC']:.4f}, "
          f"Balanced_Accuracy={best['Balanced_Accuracy']:.4f}, Thresh={best['Thresh']:.3f}")
    with open(out/"best_model_summary.json","w") as f:
        json.dump({"criteria": why, "best": best}, f, indent=2)

    if xgb_available and "RF" in preds and "XGB" in preds:
        disagree = int(np.sum(preds["RF"] != preds["XGB"]))
        print(f"[Check] RF vs XGB label disagreement on test: {disagree} / {len(preds['RF'])}")

    roc_png = out/"roc_two_models.png"
    plot_roc(model_probs, yte, str(roc_png))
    print(f"ROC saved to: {roc_png}")

    rf_png, rf_csv, rf_method = shap_rf(rf_pipe, Xte, features, str(out/"rf"),
                                        max_n=200, show_progress=SHOW_PROGRESS)
    print(f"RF importance saved: {rf_png}, {rf_csv} ({rf_method})")

    if xgb_available:
        xgb_png, xgb_csv = shap_xgb_native(xgb_pipe, Xte, features, str(out/"xgb"),
                                           max_n=500, show_progress=SHOW_PROGRESS)
        if xgb_png:
            print(f"XGB TreeSHAP saved: {xgb_png}, {xgb_csv}")

    print(f"[Done] Total time: {(time.time()-t0):.1f}s")
