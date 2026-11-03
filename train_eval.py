"""
train_eval.py - ocena modeli metoda walk-forward (bez zagladania w przyszlosc).

Modele (wybrane w poprzednich testach):
  LR    - regresja logistyczna -> szansa wygranej gospodarza
  Ridge - regresja grzbietowa   -> roznica punktow
Dane treningowe: 8 sezonow przed sezonem testowym.

Regularyzacja stala: C=0.01 (LR), alpha=100 (Ridge). Dobieranie jej na jednym
sezonie walidacyjnym dawalo niestabilne wartosci i nie poprawialo wyniku.

Wejscie: data/features.csv (z build_features.py)
Wyjscie: tabele w terminalu + data/results.csv

Uruchom:
    python3 train_eval.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, log_loss, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURES_CSV = "data/features.csv"
RESULTS_CSV = "data/results.csv"
FIRST_TRAIN_SEASON = 2011        # 2010 nie ma poprzedniego sezonu (brak cech prev_*)
TEST_SEASONS = [2022, 2023, 2024, 2025, 2026, 2027]   # 2022 = kontrola z pracy, 2027 = biezacy
EVAL_SEASONS = [2023, 2024, 2025, 2026]               # do usredniania
TRAIN_YEARS = 8
EXTRA = ["hfa_prior", "elo_prob"]                      # cechy meczu (nie roznice)
C = 0.01
ALPHA = 100

# grupy cech do ablacji
ELO = ["diff_elo", "elo_prob", "hfa_prior"]
SKLAD = ["diff_lineup_rating", "diff_missing_rating"]


def lr_model(c):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=c, max_iter=2000))


def ridge_model(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Ridge(alpha=alpha))


def train_rows(df, test):
    return df[(df["season"] >= max(FIRST_TRAIN_SEASON, test - TRAIN_YEARS)) &
              (df["season"] < test)]


def fit_models(df, cols, test):
    """Modele uczone na 8 sezonach przed sezonem testowym."""
    tr = train_rows(df, test)
    lr = lr_model(C).fit(tr[cols], tr["home_win"])
    ridge = ridge_model(ALPHA).fit(tr[cols], tr["margin"])
    return lr, ridge, len(tr)


def test_rows(df, test):
    te = df[df["season"] == test]
    return te[te["market_margin"].notna()] if test < 2027 else te


def evaluate(df, cols, test):
    lr, ridge, n = fit_models(df, cols, test)
    te = test_rows(df, test)
    p = lr.predict_proba(te[cols])[:, 1]
    m = ridge.predict(te[cols])
    r = {"sezon": test, "n_train": n,
         "acc": accuracy_score(te["home_win"], p > 0.5),
         "auc": roc_auc_score(te["home_win"], p),
         "logloss": log_loss(te["home_win"], p),
         "mae": mean_absolute_error(te["margin"], m),
         "rmse": mean_squared_error(te["margin"], m) ** 0.5}
    return r, te.assign(p_home=p, pred_margin=m)


def main():
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=["home_win", "margin"])
    feats = [c for c in df.columns if c.startswith("diff_")] + [c for c in EXTRA if c in df]
    print(f"Mecze: {len(df)}, cechy: {len(feats)}")

    rows, preds, market = [], [], []
    for test in TEST_SEASONS:
        r, te = evaluate(df, feats, test)
        rows.append(r)
        preds.append(te)
        mk = te[te["market_margin"].notna() & (te["market_margin"] != 0)]
        if len(mk):
            market.append({
                "sezon": test, "mecze": len(te),
                "acc_rynek": ((mk["market_margin"] > 0) == (mk["home_win"] == 1)).mean(),
                "mae_rynek": (te["margin"] - te["market_margin"]).abs().mean(),
            })
        print(f"sezon {test}: gotowe")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_CSV, index=False)
    mk = pd.DataFrame(market)
    pd.set_option("display.width", 200)

    print("\n=== Wyniki per sezon (LR: acc/auc/logloss, Ridge: mae/rmse) ===")
    print(res.round(3).to_string(index=False))
    print("\n=== Rynek ===")
    print(mk.round(3).to_string(index=False))

    ev = res[res["sezon"].isin(EVAL_SEASONS)]
    m = mk[mk["sezon"].isin(EVAL_SEASONS)]
    print(f"\nSrednia 2023-2026  model: acc={ev['acc'].mean():.3f}  logloss={ev['logloss'].mean():.3f}"
          f"  MAE={ev['mae'].mean():.3f}")
    print(f"Srednia 2023-2026  rynek: acc={m['acc_rynek'].mean():.3f}  MAE={m['mae_rynek'].mean():.3f}")

    # kalibracja: czy "60%" oznacza, ze gospodarz wygrywa w ok. 60% przypadkow
    allp = pd.concat([p for p in preds if p["season"].iloc[0] in EVAL_SEASONS])
    bins = pd.cut(allp["p_home"], np.arange(0, 1.01, 0.1))
    cal = allp.groupby(bins, observed=True).agg(
        mecze=("home_win", "size"), przewidywane=("p_home", "mean"), faktyczne=("home_win", "mean"))
    print("\n=== Kalibracja LR (2023-2026) ===")
    print(cal.round(3).to_string())
    ece = (cal["mecze"] * (cal["przewidywane"] - cal["faktyczne"]).abs()).sum() / cal["mecze"].sum()
    print(f"Sredni blad kalibracji: {ece:.3f}")

    # ablacja: ktore grupy cech pomagaja
    base = [f for f in feats if f not in ELO + SKLAD]
    sets = {"bazowe": base, "+elo": base + ELO, "+sklad": base + SKLAD, "wszystkie": feats}
    out = []
    for name, cols in sets.items():
        cols = [c for c in cols if c in df]
        for test in EVAL_SEASONS:
            r, _ = evaluate(df, cols, test)
            out.append({"zestaw": name, **r})
    abl = pd.DataFrame(out).groupby("zestaw", sort=False)[["acc", "auc", "logloss", "mae"]].mean()
    print("\n=== Ablacja (2023-2026) ===")
    print(abl.round(3).to_string())
    # najwazniejsze cechy modelu koncowego (8 sezonow przed biezacym)
    lr, ridge, _ = fit_models(df, feats, max(TEST_SEASONS))
    imp = pd.DataFrame({"LR": lr[-1].coef_[0], "Ridge_pkt": ridge[-1].coef_}, index=feats)
    imp = imp.reindex(imp["LR"].abs().sort_values(ascending=False).index)
    print("\n=== Najwazniejsze cechy (model koncowy, efekt zmiany o 1 odch. std.) ===")
    print(imp.round(3).head(15).to_string())
    print(f"\nPelne wyniki: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
