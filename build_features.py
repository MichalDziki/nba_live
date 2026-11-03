"""
build_features.py - buduje tabele cech (1 wiersz = 1 mecz sezonu zasadniczego).

Wejscie (z get_data.py i Kaggle):
  data/nba/team_game_logs_<sezon>.parquet
  data/nba/player_game_logs_<sezon>.parquet
  data/nba_2008-2026.csv                      (kursy, opcjonalnie)

Wyjscie:
  data/features.csv

Uruchom:
    python3 build_features.py

Cechy (dla gospodarza home_*, goscia away_* i roznica diff_*):
  roll_*  - srednia z 10 poprzednich meczow w sezonie (shift(1): bez biezacego meczu)
  prev_*  - srednia z calego poprzedniego sezonu
  rest_days, b2b, games_played - kontekst meczu
  missing_min, missing_pts     - utracone minuty/punkty: zawodnicy, ktorzy grali
                                 w ostatnich 10 meczach druzyny, a w tym meczu nie zagrali
  missing_core_min/_n          - to samo, tylko zawodnicy rotacji (srednio >= 20 min)
  lineup_rating                - sila skladu: srednia ocen OpenRAPM (z poprzedniego sezonu)
                                 grajacych zawodnikow wazona spodziewanymi minutami, x5
  missing_rating               - ocena x minuty/48 zawodnikow nieobecnych
  elo                          - rating Elo druzyny przed meczem; elo_prob - szansa
                                 gospodarza z Elo; hfa_prior - przewaga gospodarza z 3
                                 poprzednich sezonow
Cele: home_win, margin (pkt gospodarza - pkt goscia), total
Rynek: market_margin (z spreadu), market_p_home (z moneyline, bez marzy)
"""
import glob
import os
import re

import numpy as np
import pandas as pd

DATA = "data"
NBA = os.path.join(DATA, "nba")
ODDS = os.path.join(DATA, "nba_2008-2026.csv")
OUT = os.path.join(DATA, "features.csv")
WINDOW = 10
CORE_MIN = 20   # zawodnik "rotacji": srednio min. 20 minut w grze w ostatnich 10 meczach
ROT_MIN = 10    # ponizej tego (srednio) zawodnik gra glownie w garbage time - pomijamy go,
                # bo to, czy wejdzie na boisko, zalezy od wyniku meczu (przeciek)
REPLACEMENT = -2.0  # ocena zawodnika bez oceny z poprzedniego sezonu (debiutant itp.)
NEW_PLAYER_MIN = 15  # spodziewane minuty zawodnika bez zadnej historii
# Elo: K i powrot do sredniej miedzy sezonami (siatka 2011-2021 wybrala te same wartosci).
# Przewaga gospodarza w danym sezonie = srednia roznica pkt gospodarzy z 3 poprzednich sezonow.
ELO_K, ELO_CARRY = 20, 0.75
ELO_PER_POINT = 28      # ok. 28 punktow Elo = 1 punkt roznicy w meczu
HFA_DEFAULT = 2.5       # przewaga gospodarza w pkt, gdy brak poprzednich sezonow

# Statystyki druzyny w meczu, z ktorych liczymy forme (uzywamy tych, ktore istnieja)
STATS = ["win", "plus_minus", "pts", "opp_pts", "off_rating", "def_rating",
         "net_rating", "pace", "efg_pct", "ts_pct", "tm_tov_pct", "oreb_pct"]

# Skroty z pliku kursow -> stale id franczyz NBA
ODDS_TEAM_ID = {
    "atl": "1610612737", "bos": "1610612738", "cle": "1610612739", "no": "1610612740",
    "chi": "1610612741", "dal": "1610612742", "den": "1610612743", "gs": "1610612744",
    "hou": "1610612745", "lac": "1610612746", "lal": "1610612747", "mia": "1610612748",
    "mil": "1610612749", "min": "1610612750", "bkn": "1610612751", "ny": "1610612752",
    "orl": "1610612753", "ind": "1610612754", "phi": "1610612755", "phx": "1610612756",
    "por": "1610612757", "sac": "1610612758", "sa": "1610612759", "okc": "1610612760",
    "tor": "1610612761", "utah": "1610612762", "mem": "1610612763", "wsh": "1610612764",
    "det": "1610612765", "cha": "1610612766",
}


def load(kind: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(NBA, f"{kind}_*.parquet")))
    if not files:
        raise SystemExit(f"Brak plikow {kind}_*.parquet w {NBA}. Uruchom najpierw get_data.py")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if "season" not in df.columns:
            df["season"] = int(re.search(r"_(\d{4})\.parquet$", f).group(1))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in ("game_id", "team_id", "opp_team_id", "player_id"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    df["season"] = df["season"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    # tylko sezon zasadniczy: id meczu zaczyna sie od "002"
    return df[df["game_id"].str.startswith("002")].copy()


# ---------------------------------------------------------------- druzyny
def team_features(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.copy()
    tg["win"] = tg["win"].astype(float)
    tg["opp_pts"] = tg["pts"] - tg["plus_minus"]
    stats = [s for s in STATS if s in tg.columns]
    missing = sorted(set(STATS) - set(stats))
    if missing:
        print(f"Uwaga: brak kolumn {missing} - pomijam je")

    tg = tg.sort_values(["team_id", "game_date", "game_id"])
    g = tg.groupby(["season", "team_id"], sort=False)

    # forma: srednia z 10 poprzednich meczow (bez biezacego)
    for s in stats:
        tg[f"roll_{s}"] = g[s].transform(
            lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())

    # poprzedni sezon: srednie sezonowe przesuniete o rok
    season_avg = tg.groupby(["season", "team_id"])[stats].mean()
    season_avg.columns = [f"prev_{s}" for s in stats]
    season_avg = season_avg.reset_index()
    season_avg["season"] += 1
    tg = tg.merge(season_avg, on=["season", "team_id"], how="left")

    # na poczatku sezonu (brak poprzednich meczow) forma = poprzedni sezon
    for s in stats:
        tg[f"roll_{s}"] = tg[f"roll_{s}"].fillna(tg[f"prev_{s}"])

    # kontekst
    tg = tg.sort_values(["team_id", "game_date", "game_id"])
    g = tg.groupby(["season", "team_id"], sort=False)
    days = g["game_date"].diff().dt.days
    tg["rest_days"] = days.clip(upper=7).fillna(7)
    tg["b2b"] = (days == 1).astype(int)
    tg["games_played"] = g.cumcount()
    return tg


# ---------------------------------------------------------------- kontuzje
def load_ratings() -> dict:
    """Oceny OpenRAPM: (sezon, player_id) -> rating. Brak plikow = pusty slownik."""
    files = sorted(glob.glob(os.path.join(NBA, "openrapm_*.parquet")))
    if not files:
        print("Uwaga: brak openrapm_*.parquet - sila skladu bez ocen zawodnikow")
        return {}
    r = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    r["player_id"] = r["player_id"].astype(str)
    return {(int(s), p): float(v) for s, p, v in zip(r["season"], r["player_id"], r["rating"])}


def missing_minutes(pg: pd.DataFrame, tg: pd.DataFrame, ratings: dict) -> pd.DataFrame:
    """Dla kazdego meczu druzyny: suma srednich minut/punktow zawodnikow,
    ktorzy grali w ostatnich 10 meczach tej druzyny, a w tym meczu nie zagrali.
    Zawodnik liczy sie tylko, jesli nadal nalezy do druzyny (ostatni mecz przed
    i najblizszy mecz od tej daty rozegral dla tej druzyny), zeby nie liczyc
    zawodnikow po wymianie jako kontuzjowanych."""
    pg = pg.copy()
    pg["min"] = pd.to_numeric(pg["min"], errors="coerce")
    pg = pg[pg["min"] > 0]
    out = []
    # srednie minuty na mecz w calym sezonie (dla zawodnika, niezaleznie od druzyny)
    mpg = pg.groupby(["season", "player_id"])["min"].mean().to_dict()

    for season, ps in pg.groupby("season"):
        # historia kazdego zawodnika w sezonie (data, druzyna) - do sprawdzania skladu
        hist = {pid: (h["game_date"].values, h["team_id"].values)
                for pid, h in ps.sort_values("game_date").groupby("player_id")}
        ts = tg[tg["season"] == season]

        for team, games in ts.groupby("team_id"):
            games = games.sort_values(["game_date", "game_id"])
            gids = games["game_id"].values
            dates = games["game_date"].values
            rows = ps[ps["team_id"] == team]
            if rows.empty:
                continue
            m = rows.pivot_table(index="game_id", columns="player_id",
                                 values="min", aggfunc="sum").reindex(gids)
            p = rows.pivot_table(index="game_id", columns="player_id",
                                 values="pts", aggfunc="sum").reindex(gids)

            played = m.notna()
            cnt = played.astype(float).shift(1).rolling(WINDOW, min_periods=1).sum()
            avg_min = m.fillna(0).shift(1).rolling(WINDOW, min_periods=1).sum() / cnt
            avg_pts = p.fillna(0).shift(1).rolling(WINDOW, min_periods=1).sum() / cnt

            # czy zawodnik nalezy do druzyny w dniu meczu
            roster = pd.DataFrame(False, index=m.index, columns=m.columns)
            for pid in m.columns:
                d_all, t_all = hist[pid]
                i = np.searchsorted(d_all, dates, side="left")
                prev_ok = (i > 0) & (t_all[np.maximum(i - 1, 0)] == team)
                nxt = np.minimum(i, len(d_all) - 1)
                next_ok = (i >= len(d_all)) | (t_all[nxt] == team)
                roster[pid] = prev_ok & next_ok

            absent = (cnt > 0) & (~played) & roster & (avg_min >= ROT_MIN)
            core = absent & (avg_min >= CORE_MIN)   # tylko zawodnicy rotacji

            # Sila skladu: ocena z POPRZEDNIEGO sezonu (bez przecieku z biezacego),
            # wazona spodziewanymi minutami zawodnikow, ktorzy graja w tym meczu.
            # Spodziewane minuty: srednia z ostatnich 10 meczow, a gdy brak
            # (poczatek sezonu, powrot po kontuzji) - srednia z poprzedniego sezonu.
            rating = pd.Series({pid: ratings.get((season - 1, pid), REPLACEMENT)
                                for pid in m.columns})
            prev_mpg = pd.Series({pid: mpg.get((season - 1, pid), NEW_PLAYER_MIN)
                                  for pid in m.columns})
            exp_min = avg_min.where(cnt > 0).fillna(prev_mpg).clip(upper=40)
            w = exp_min.where(played & (exp_min >= ROT_MIN))
            lineup = 5 * (w * rating).sum(axis=1) / w.sum(axis=1)
            lost = (avg_min.where(absent) * rating).sum(axis=1) / 48

            out.append(pd.DataFrame({
                "game_id": gids,
                "team_id": team,
                "missing_min": avg_min.where(absent).sum(axis=1).values,
                "missing_pts": avg_pts.where(absent).sum(axis=1).values,
                "missing_core_min": avg_min.where(core).sum(axis=1).values,
                "missing_core_n": core.sum(axis=1).values,
                "lineup_rating": lineup.values,
                "missing_rating": lost.values,
            }))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- mecze
def game_table(tg: pd.DataFrame) -> pd.DataFrame:
    feat = [c for c in tg.columns if c.startswith(("roll_", "prev_"))] + \
           ["rest_days", "b2b", "games_played", "missing_min", "missing_pts",
            "missing_core_min", "missing_core_n", "lineup_rating", "missing_rating"]
    neutral = tg["home"].isna().groupby(tg["game_id"]).any()
    if neutral.sum():
        print(f"Pomijam {int(neutral.sum())} meczow na neutralnym terenie")
    tg = tg[~tg["game_id"].isin(neutral[neutral].index)]

    home = tg[tg["home"] == True]   # noqa: E712
    away = tg[tg["home"] == False]  # noqa: E712
    keep = ["game_id", "season", "game_date", "team_id", "pts"] + feat
    h = home[keep].rename(columns={c: f"home_{c}" for c in keep[3:]})
    a = away[keep].drop(columns=["season", "game_date"]) \
        .rename(columns={c: f"away_{c}" for c in keep[3:]})
    games = h.merge(a, on="game_id", how="inner")

    for c in feat:
        games[f"diff_{c}"] = games[f"home_{c}"] - games[f"away_{c}"]
    games["margin"] = games["home_pts"] - games["away_pts"]
    games["total"] = games["home_pts"] + games["away_pts"]
    games["home_win"] = (games["margin"] > 0).astype(int)
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- Elo
def run_elo(games: pd.DataFrame, k: float, carry: float):
    """Elo przed meczem (tylko z wczesniejszych meczow). Mecze musza byc
    posortowane po dacie. Miedzy sezonami Elo wraca czesciowo do sredniej."""
    elo, last = {}, {}
    pre_h, pre_a, prob = [], [], []
    for s, h, a, mg, hfa in zip(games["season"], games["home_team_id"],
                                games["away_team_id"], games["margin"], games["hfa_prior"]):
        for t in (h, a):
            if t not in elo:
                elo[t] = 1500.0
            elif last[t] != s:
                elo[t] = carry * elo[t] + (1 - carry) * 1505
            last[t] = s
        eh, ea = elo[h], elo[a]
        d = eh + ELO_PER_POINT * hfa - ea
        exp_h = 1 / (1 + 10 ** (-d / 400))
        pre_h.append(eh)
        pre_a.append(ea)
        prob.append(exp_h)
        if pd.isna(mg):
            continue
        win = 1.0 if mg > 0 else 0.0
        d_winner = d if win else -d
        mult = (abs(mg) + 3) ** 0.8 / (7.5 + 0.006 * d_winner)
        delta = k * mult * (win - exp_h)
        elo[h] += delta
        elo[a] -= delta
    return np.array(pre_h), np.array(pre_a), np.array(prob)


def add_elo(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    # przewaga gospodarza w sezonie = srednia roznica pkt z 3 poprzednich sezonow
    by_season = games.groupby("season")["margin"].mean()
    hfa = {}
    for s in games["season"].unique():
        prev = by_season[(by_season.index >= s - 3) & (by_season.index < s)]
        hfa[s] = prev.mean() if len(prev) else HFA_DEFAULT
    games["hfa_prior"] = games["season"].map(hfa)

    h, a, p = run_elo(games, ELO_K, ELO_CARRY)
    games["home_elo"] = h
    games["away_elo"] = a
    games["diff_elo"] = h - a
    games["elo_prob"] = p
    return games


# ---------------------------------------------------------------- kursy
def american_to_prob(ml: pd.Series) -> pd.Series:
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def add_odds(games: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(ODDS):
        print(f"Brak {ODDS} - pomijam kursy")
        return games
    o = pd.read_csv(ODDS)
    o = o[o["regular"] == True].copy()  # noqa: E712
    o["home_team_id"] = o["home"].map(ODDS_TEAM_ID)
    o["away_team_id"] = o["away"].map(ODDS_TEAM_ID)
    o["game_date"] = pd.to_datetime(o["date"])

    # spread bez znaku + faworyt -> oczekiwana roznica punktow gospodarza wg rynku
    sign = np.where(o["whos_favored"] == "home", 1, -1)
    o["market_margin"] = sign * pd.to_numeric(o["spread"], errors="coerce")
    o["market_total"] = pd.to_numeric(o["total"], errors="coerce")
    ph = american_to_prob(o["moneyline_home"])
    pa = american_to_prob(o["moneyline_away"])
    o["market_p_home"] = ph / (ph + pa)            # bez marzy bukmachera
    o["ml_home"] = pd.to_numeric(o["moneyline_home"], errors="coerce")
    o["ml_away"] = pd.to_numeric(o["moneyline_away"], errors="coerce")
    o["odds_home_pts"] = o["score_home"]
    o["odds_away_pts"] = o["score_away"]
    cols = ["home_team_id", "away_team_id", "game_date", "market_margin",
            "market_total", "market_p_home", "ml_home", "ml_away",
            "odds_home_pts", "odds_away_pts"]
    o = o[cols].drop_duplicates(["home_team_id", "away_team_id", "game_date"])

    merged = games.merge(o, on=["home_team_id", "away_team_id", "game_date"], how="left")

    # mecze bez dopasowania: proba z data +-1 dzien (rozne strefy czasowe)
    miss = merged["market_margin"].isna() & merged["odds_home_pts"].isna()
    for shift in (-1, 1):
        if not miss.any():
            break
        o2 = o.copy()
        o2["game_date"] = o2["game_date"] + pd.Timedelta(days=shift)
        fix = merged.loc[miss, ["game_id", "home_team_id", "away_team_id", "game_date"]] \
            .merge(o2, on=["home_team_id", "away_team_id", "game_date"], how="inner")
        fix = fix.drop_duplicates("game_id").set_index("game_id")
        idx = merged["game_id"].isin(fix.index) & miss
        for c in cols[3:]:
            merged.loc[idx, c] = merged.loc[idx, "game_id"].map(fix[c])
        miss = merged["market_margin"].isna() & merged["odds_home_pts"].isna()

    # kontrola: czy wyniki z obu zrodel sa takie same
    both = merged["odds_home_pts"].notna()
    bad = both & ((merged["odds_home_pts"] != merged["home_pts"]) |
                  (merged["odds_away_pts"] != merged["away_pts"]))
    print(f"Kursy: dopasowano {int(both.sum())} z {len(merged)} meczow; "
          f"niezgodny wynik w {int(bad.sum())}")
    merged.loc[bad, ["market_margin", "market_total", "market_p_home", "ml_home", "ml_away"]] = np.nan
    return merged.drop(columns=["odds_home_pts", "odds_away_pts"])


# ---------------------------------------------------------------- main
def main() -> None:
    tg = load("team_game_logs")
    pg = load("player_game_logs")
    print(f"Wczytano: {len(tg)} wierszy druzyn, {len(pg)} wierszy zawodnikow, "
          f"sezony {tg['season'].min()}-{tg['season'].max()}")

    tg = team_features(tg)
    ratings = load_ratings()
    mm = missing_minutes(pg, tg, ratings)
    tg = tg.merge(mm, on=["game_id", "team_id"], how="left")
    miss_cols = ["missing_min", "missing_pts", "missing_core_min", "missing_core_n",
                 "missing_rating"]
    tg[miss_cols] = tg[miss_cols].fillna(0)

    games = game_table(tg)
    games = add_elo(games)
    games = add_odds(games)
    games.to_csv(OUT, index=False)

    # podsumowanie do wklejenia
    print(f"\nZapisano {OUT}: {len(games)} meczow, {games.shape[1]} kolumn")
    summary = games.groupby("season").agg(
        mecze=("game_id", "size"),
        home_win=("home_win", "mean"),
        margin=("margin", "mean"),
        brak_prev=("home_prev_win", lambda s: int(s.isna().sum())),
        missing_min=("home_missing_min", "mean"),
        core_min=("home_missing_core_min", "mean"),
        core_n=("home_missing_core_n", "mean"),
        lineup=("home_lineup_rating", "mean"),
        elo_sd=("home_elo", "std"),
        spread=("market_margin", lambda s: int(s.notna().sum())) if "market_margin" in games else ("game_id", "size"),
        moneyline=("market_p_home", lambda s: int(s.notna().sum())) if "market_p_home" in games else ("game_id", "size"),
    ).round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
