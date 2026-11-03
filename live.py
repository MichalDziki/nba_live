"""
live.py - cechy i predykcje dla meczow, ktore jeszcze sie nie odbyly.
Uzywane przez app.py. Logika cech jest ta sama co w build_features.py
(sprawdzane testem zgodnosci na meczach historycznych).
"""
import datetime as dt
import glob
import os
import re
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import build_features as bf
from train_eval import ALPHA, C, EXTRA, FEATURES_CSV, FIRST_TRAIN_SEASON, TRAIN_YEARS, \
    lr_model, ridge_model

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
TEAM_ID = {k.upper(): v for k, v in bf.ODDS_TEAM_ID.items()}   # skroty ESPN -> id NBA
TEAM_ABBR = {v: k for k, v in TEAM_ID.items()}
PAPER_CSV = os.path.join(bf.DATA, "paper_bets.csv")
NBA_INJURY_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{t}.pdf"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
# statusy ESPN -> statusy raportu NBA ("Day-To-Day" traktujemy jak "Questionable")
ESPN_STATUS = {"out": "Out", "doubtful": "Doubtful", "questionable": "Questionable",
               "day-to-day": "Questionable", "probable": "Probable", "suspension": "Out",
               "available": "Available"}
STATUSES = ["Out", "Doubtful", "Questionable", "Probable", "Available"]


def season_of(date: dt.date) -> int:
    """Sezon oznaczony rokiem koncowym (pazdziernik 2026 -> 2027)."""
    return date.year + 1 if date.month >= 8 else date.year


# ---------------------------------------------------------------- terminarz
def fetch_schedule(date: dt.date) -> pd.DataFrame:
    r = requests.get(ESPN_SCOREBOARD, params={"dates": date.strftime("%Y%m%d")},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = []
    for ev in r.json().get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c["team"]["abbreviation"].upper() for c in comp["competitors"]}
        h, a = teams.get("home"), teams.get("away")
        if h not in TEAM_ID or a not in TEAM_ID:
            continue
        rows.append({"espn_id": ev["id"], "start_utc": ev.get("date"), "home": h, "away": a,
                     "home_team_id": TEAM_ID[h], "away_team_id": TEAM_ID[a],
                     "status": comp["status"]["type"].get("description", "")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- uzupelnianie z ESPN
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
MAX_GAP_DAYS = 10


def _made_att(v):
    try:
        m, a = str(v).split("-")
        return float(m), float(a)
    except ValueError:
        return np.nan, np.nan


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def espn_day(date: dt.date) -> list:
    """Zakonczone mecze danego dnia z ESPN: lista slownikow z wynikiem i box score."""
    r = requests.get(ESPN_SCOREBOARD, params={"dates": date.strftime("%Y%m%d")},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    out = []
    for ev in r.json().get("events", []):
        comp = ev["competitions"][0]
        if not comp["status"]["type"].get("completed"):
            continue
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        h = teams["home"]["team"]["abbreviation"].upper()
        a = teams["away"]["team"]["abbreviation"].upper()
        if h not in TEAM_ID or a not in TEAM_ID:
            continue
        sm = requests.get(ESPN_SUMMARY, params={"event": ev["id"]}, headers=HEADERS, timeout=30)
        sm.raise_for_status()
        out.append({"espn_id": ev["id"], "date": date, "home": TEAM_ID[h], "away": TEAM_ID[a],
                    "home_pts": _num(teams["home"].get("score")),
                    "away_pts": _num(teams["away"].get("score")),
                    "box": sm.json().get("boxscore", {})})
    return out


def espn_to_logs(games: list, season: int, names: dict) -> tuple:
    """Box score'y ESPN -> wiersze w formacie team_game_logs i player_game_logs (id NBA).
    Statystyki zaawansowane liczone wzorami z box score'u."""
    index = {squash(n): pid for pid, n in names.items() if n}
    team_rows, player_rows, unmatched = [], [], []
    for g in games:
        gid = f"ESPN_{g['espn_id']}"
        box = {}
        for t in g["box"].get("teams", []):
            st_ = {x.get("name"): x.get("displayValue") for x in t.get("statistics", [])}
            fgm, fga = _made_att(st_.get("fieldGoalsMade-fieldGoalsAttempted"))
            f3m, f3a = _made_att(st_.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
            ftm, fta = _made_att(st_.get("freeThrowsMade-freeThrowsAttempted"))
            tov = _num(st_.get("totalTurnovers", st_.get("turnovers")))
            oreb, dreb = _num(st_.get("offensiveRebounds")), _num(st_.get("defensiveRebounds"))
            box[t.get("homeAway")] = dict(fgm=fgm, fga=fga, fg3m=f3m, fg3a=f3a, ftm=ftm, fta=fta,
                                          oreb=oreb, dreb=dreb, reb=oreb + dreb,
                                          ast=_num(st_.get("assists")), pf=_num(st_.get("fouls")),
                                          tov=tov)
        if set(box) != {"home", "away"}:
            continue
        poss = {k: b["fga"] + 0.44 * b["fta"] - b["oreb"] + b["tov"] for k, b in box.items()}
        p_avg = (poss["home"] + poss["away"]) / 2
        for side, other, team, opp, home in (("home", "away", g["home"], g["away"], True),
                                             ("away", "home", g["away"], g["home"], False)):
            b, o = box[side], box[other]
            pts, opp_pts = g[f"{side}_pts"], g[f"{other}_pts"]
            team_rows.append({
                "season": season, "game_id": gid, "team_id": team, "opp_team_id": opp,
                "game_date": pd.Timestamp(g["date"]), "home": home, "win": pts > opp_pts,
                "pts": pts, "plus_minus": pts - opp_pts, **b,
                "off_rating": 100 * pts / p_avg, "def_rating": 100 * opp_pts / p_avg,
                "net_rating": 100 * (pts - opp_pts) / p_avg, "pace": p_avg,
                "efg_pct": (b["fgm"] + 0.5 * b["fg3m"]) / b["fga"],
                "ts_pct": pts / (2 * (b["fga"] + 0.44 * b["fta"])),
                "tm_tov_pct": 100 * b["tov"] / (b["fga"] + 0.44 * b["fta"] + b["tov"]),
                "oreb_pct": b["oreb"] / (b["oreb"] + o["dreb"]),
                "source": "espn",
            })
        abbr_side = {}
        for t in g["box"].get("teams", []):
            abbr_side[t.get("team", {}).get("abbreviation", "").upper()] = t.get("homeAway")
        for t in g["box"].get("players", []):
            side = abbr_side.get(t.get("team", {}).get("abbreviation", "").upper())
            team = g.get(side)
            if team is None:
                continue
            for grp in t.get("statistics", []):
                cols = grp.get("names", [])
                for a in grp.get("athletes", []):
                    if a.get("didNotPlay"):
                        continue
                    vals = dict(zip(cols, a.get("stats", [])))
                    mins = _num(vals.get("MIN"))
                    if not mins or np.isnan(mins) or mins <= 0:
                        continue
                    name = a.get("athlete", {}).get("displayName", "")
                    pid = index.get(squash(name))
                    if pid is None:
                        unmatched.append(name)
                        continue
                    fgm_, fga_ = _made_att(vals.get("FG"))
                    f3m_, f3a_ = _made_att(vals.get("3PT"))
                    ftm_, fta_ = _made_att(vals.get("FT"))
                    player_rows.append({"season": season, "game_id": gid, "player_id": pid,
                                        "team_id": team, "game_date": pd.Timestamp(g["date"]),
                                        "min": mins, "pts": _num(vals.get("PTS")),
                                        "reb": _num(vals.get("REB")), "ast": _num(vals.get("AST")),
                                        "stl": _num(vals.get("STL")), "blk": _num(vals.get("BLK")),
                                        "tov": _num(vals.get("TO")), "pf": _num(vals.get("PF")),
                                        "oreb": _num(vals.get("OREB")), "dreb": _num(vals.get("DREB")),
                                        "fgm": fgm_, "fga": fga_, "fg3m": f3m_, "fg3a": f3a_,
                                        "ftm": ftm_, "fta": fta_,
                                        "plus_minus": _num(vals.get("+/-"))})
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows), sorted(set(unmatched))


# ---------------------------------------------------------------- dane
def load_names(seasons) -> dict:
    names = {}
    for s in seasons:
        f = os.path.join(bf.NBA, f"player_seasons_{s}.parquet")
        if os.path.exists(f):
            d = pd.read_parquet(f)
            names.update(dict(zip(d["player_id"].astype(str), d["name"].fillna(""))))
    return names


class History:
    """Dane potrzebne do cech: 2 ostatnie sezony logow + caly features.csv (Elo, trening)."""

    def __init__(self, season: int, fill_from_espn: bool = True):
        self.season = season
        tg = bf.load("team_game_logs")
        pg = bf.load("player_game_logs")
        self.tg = tg[tg["season"].isin([season - 1, season])].copy()
        pg = pg[pg["season"].isin([season - 1, season])].copy()
        pg["min"] = pd.to_numeric(pg["min"], errors="coerce")
        self.pg = pg[pg["min"] > 0].copy()
        self.ratings = bf.load_ratings()
        self.names = load_names([season - 1, season])
        self.mpg_prev = self.pg[self.pg["season"] == season - 1] \
            .groupby("player_id")["min"].mean().to_dict()
        self.features = pd.read_csv(FEATURES_CSV, dtype={
            "game_id": str, "home_team_id": str, "away_team_id": str})
        self.features["game_date"] = pd.to_datetime(self.features["game_date"])
        self.last_nba = self.tg["game_date"].max()
        self.gap_games = pd.DataFrame(columns=["season", "game_date", "home_team_id",
                                               "away_team_id", "margin"])
        self.gap_info, self.gap_unmatched, self.espn_check = "", [], None
        if fill_from_espn:
            self.fill_gap()
        self.add_per()

    def add_per(self):
        """PER (Hollinger, jak na Basketball Reference) - licznik dla kazdego meczu zawodnika.
        PER z kilku meczow = 15 * suma(per_num) / suma(minut) / srednia ligowa."""
        stat = ["fgm", "fga", "fg3m", "ftm", "fta", "oreb", "reb", "ast", "stl", "blk", "tov",
                "pf", "pts"]
        pg = self.pg.reset_index(drop=True)
        tg = self.tg.copy()
        for c in stat:
            pg[c] = pd.to_numeric(pg[c], errors="coerce") if c in pg else np.nan
            tg[c] = pd.to_numeric(tg[c], errors="coerce") if c in tg else np.nan
        team = tg[["game_id", "team_id", "ast", "fgm", "pace"]].rename(
            columns={"ast": "tm_ast", "fgm": "tm_fg", "pace": "tm_pace"})
        d = pg.merge(team, on=["game_id", "team_id"], how="left")
        d["per_num"] = np.nan
        self.lg_per = {}
        for s, idx in d.groupby("season").groups.items():
            L = tg[tg["season"] == s][stat].sum()
            lg_pace = tg.loc[tg["season"] == s, "pace"].mean()
            if not (L["fgm"] > 0 and L["ftm"] > 0 and L["pf"] > 0 and L["reb"] > 0):
                continue
            factor = 2 / 3 - (0.5 * L["ast"] / L["fgm"]) / (2 * L["fgm"] / L["ftm"])
            vop = L["pts"] / (L["fga"] - L["oreb"] + L["tov"] + 0.44 * L["fta"])
            drbp = (L["reb"] - L["oreb"]) / L["reb"]
            x = d.loc[idx]
            r = x["tm_ast"] / x["tm_fg"]
            u = (x["fg3m"] + 2 / 3 * x["ast"] + (2 - factor * r) * x["fgm"]
                 + x["ftm"] * 0.5 * (1 + (1 - r) + 2 / 3 * r)
                 - vop * x["tov"] - vop * drbp * (x["fga"] - x["fgm"])
                 - vop * 0.44 * (0.44 + 0.56 * drbp) * (x["fta"] - x["ftm"])
                 + vop * (1 - drbp) * (x["reb"] - x["oreb"]) + vop * drbp * x["oreb"]
                 + vop * x["stl"] + vop * drbp * x["blk"]
                 - x["pf"] * (L["ftm"] / L["pf"] - 0.44 * (L["fta"] / L["pf"]) * vop))
            a = u * lg_pace / x["tm_pace"]
            d.loc[idx, "per_num"] = a
            ok = a.notna() & x["min"].notna()
            if x.loc[ok, "min"].sum() > 0:
                self.lg_per[s] = a[ok].sum() / x.loc[ok, "min"].sum()
        self.pg = d.drop(columns=["tm_ast", "tm_fg", "tm_pace"])

    def fill_gap(self):
        """Dni od ostatniego meczu w nba_data do dzis (USA) dociagamy z ESPN."""
        today = dt.datetime.now(ZoneInfo("America/New_York")).date()
        if season_of(today) != self.season or pd.isna(self.last_nba):
            return
        # od ostatniego dnia w nba_data (moze miec tylko czesc meczow tego dnia) do dzis
        start = self.last_nba.date()
        days = [start + dt.timedelta(days=i) for i in range((today - start).days + 1)]
        if len(days) > MAX_GAP_DAYS:
            self.gap_info = f"Luka w danych ma {len(days)} dni - uzupelniam tylko ostatnie {MAX_GAP_DAYS}."
            days = days[-MAX_GAP_DAYS:]
        try:
            fetched = [g for d in days for g in espn_day(d)]
        except requests.RequestException as e:
            self.gap_info = f"Nie udalo sie uzupelnic danych z ESPN ({type(e).__name__})."
            return
        have = set(zip(self.tg["team_id"], self.tg["game_date"].dt.date))
        games = [g for g in fetched if (g["home"], g["date"]) not in have]
        check_games = [g for g in fetched if (g["home"], g["date"]) in have]
        if not games:
            self.gap_info = "Dane NBA sa aktualne - nie trzeba uzupelniac z ESPN."
        self._check(check_games)
        if not games:
            return
        tg_e, pg_e, self.gap_unmatched = espn_to_logs(games, self.season, self.names)
        self.tg = pd.concat([self.tg, tg_e], ignore_index=True)
        self.pg = pd.concat([self.pg, pg_e], ignore_index=True)
        home = tg_e[tg_e["home"] == True]  # noqa: E712
        self.gap_games = pd.DataFrame({
            "season": self.season, "game_date": home["game_date"].values,
            "home_team_id": home["team_id"].values, "away_team_id": home["opp_team_id"].values,
            "margin": home["plus_minus"].values})
        self.gap_info = (f"Uzupelniono z ESPN {len(home)} meczow "
                         f"({days[0]:%Y-%m-%d} - {tg_e['game_date'].max():%Y-%m-%d}).")

    def _check(self, check_games):
        """Kontrola: te same mecze policzone z ESPN i z oficjalnych danych NBA."""
        if not check_games:
            return
        ce, _, _ = espn_to_logs(check_games, self.season, self.names)
        off = self.tg[self.tg["game_id"].astype(str).str.startswith("00")]
        m = ce.merge(off, on=["team_id", "game_date"], suffixes=("_espn", "_nba"))
        stats = ["pts", "pace", "off_rating", "def_rating", "efg_pct", "ts_pct",
                 "tm_tov_pct", "oreb_pct"]
        self.espn_check = pd.DataFrame({
            "statystyka": stats,
            "srednia_NBA": [m[f"{c}_nba"].mean() for c in stats],
            "srednia_ESPN": [m[f"{c}_espn"].mean() for c in stats],
            "sredni_blad": [(m[f"{c}_espn"] - m[f"{c}_nba"]).abs().mean() for c in stats],
        }).round(3)

    def results(self) -> pd.DataFrame:
        """Wyniki meczow: features.csv + mecze dociagniete z ESPN."""
        cols = ["season", "game_date", "home_team_id", "away_team_id", "margin"]
        parts = [self.features[cols]] + ([self.gap_games[cols]] if len(self.gap_games) else [])
        res = pd.concat(parts, ignore_index=True)
        res["game_date"] = pd.to_datetime(res["game_date"])
        res["margin"] = pd.to_numeric(res["margin"], errors="coerce")
        return res


# ---------------------------------------------------------------- model
def feature_names(features: pd.DataFrame) -> list:
    return [c for c in features.columns if c.startswith("diff_")] + \
           [c for c in EXTRA if c in features.columns]


def train_final(features: pd.DataFrame, season: int):
    """Model koncowy: 8 poprzednich sezonow + rozegrane mecze biezacego."""
    feats = feature_names(features)
    tr = features[(features["season"] >= max(FIRST_TRAIN_SEASON, season - TRAIN_YEARS)) &
                  (features["season"] <= season)].dropna(subset=["home_win", "margin"])
    lr = lr_model(C).fit(tr[feats], tr["home_win"])
    ridge = ridge_model(ALPHA).fit(tr[feats], tr["margin"])
    return lr, ridge, feats, len(tr)


# ---------------------------------------------------------------- cechy druzyn
def team_context(h: History, games: pd.DataFrame, date: dt.date) -> dict:
    """roll_*, prev_*, rest_days, b2b, games_played dla kazdej druzyny meczu.
    Liczone funkcja z build_features na historii + 'wirtualnym' meczu."""
    ts = pd.Timestamp(date)
    hist = h.tg[h.tg["game_date"] < ts]
    virt = []
    for g in games.itertuples():
        for team, opp, home in ((g.home_team_id, g.away_team_id, True),
                                (g.away_team_id, g.home_team_id, False)):
            virt.append({"season": h.season, "game_id": f"LIVE_{g.espn_id}", "team_id": team,
                         "opp_team_id": opp, "game_date": ts, "home": home})
    tg = pd.concat([hist, pd.DataFrame(virt)], ignore_index=True)
    tg = bf.team_features(tg)
    keep = [c for c in tg.columns if c.startswith(("roll_", "prev_"))] + \
           ["rest_days", "b2b", "games_played"]
    v = tg[tg["game_id"].astype(str).str.startswith("LIVE_")]
    return {(gid, t): row[keep].to_dict() for gid, t, (_, row) in
            zip(v["game_id"], v["team_id"], v.iterrows())}


def elo_for(h: History, games: pd.DataFrame, date: dt.date) -> pd.DataFrame:
    ts = pd.Timestamp(date)
    res = h.results()
    base = res[res["game_date"] < ts]
    virt = pd.DataFrame({"season": h.season, "game_date": ts,
                         "home_team_id": games["home_team_id"].values,
                         "away_team_id": games["away_team_id"].values, "margin": np.nan,
                         "espn_id": games["espn_id"].values})
    all_g = pd.concat([base, virt], ignore_index=True).sort_values("game_date", kind="stable")
    all_g = bf.add_elo(all_g)
    return all_g[all_g["espn_id"].notna()].set_index("espn_id")


# ---------------------------------------------------------------- zawodnicy
def player_table(h: History, team: str, date: dt.date, roster_names: list = None) -> pd.DataFrame:
    """Zawodnicy druzyny przed meczem w dniu `date` (ta sama logika co w build_features).
    roster_names: aktualny sklad z ESPN (nazwiska). Jesli podany, w tabeli sa tylko ci
    zawodnicy - bez emerytow, zwolnionych i tych, ktorzy odeszli."""
    ts = pd.Timestamp(date)
    s = h.season
    cur = h.pg[(h.pg["season"] == s) & (h.pg["game_date"] < ts)]
    team_games = h.tg[(h.tg["season"] == s) & (h.tg["team_id"] == team) &
                      (h.tg["game_date"] < ts)].sort_values(["game_date", "game_id"])
    window = set(team_games["game_id"].tail(bf.WINDOW))

    # sklad: ostatni mecz zawodnika w sezonie byl dla tej druzyny
    last = cur.sort_values("game_date").groupby("player_id").tail(1)
    roster = set(last.loc[last["team_id"] == team, "player_id"])
    w = cur[(cur["team_id"] == team) & cur["game_id"].isin(window)]
    agg = w.groupby("player_id").agg(cnt=("min", "size"), avg_min=("min", "mean"),
                                     avg_pts=("pts", "mean"))
    played_for_team = set(cur.loc[cur["team_id"] == team, "player_id"])

    # zawodnicy z poprzedniego sezonu tej druzyny, ktorzy w tym sezonie jeszcze nie grali
    prev = h.pg[h.pg["season"] == s - 1].sort_values("game_date").groupby("player_id").tail(1)
    returning = set(prev.loc[prev["team_id"] == team, "player_id"]) - set(cur["player_id"])

    pids = played_for_team | returning
    extra_names = {}
    if roster_names is not None:
        def key(n):
            n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
            return re.sub(r"[^a-z0-9]", "", n)
        index = {key(n): pid for pid, n in h.names.items() if n}
        on_roster = {index[key(n)] for n in roster_names if key(n) in index}
        # zawodnicy ze skladu ESPN bez historii w danych NBA (np. debiutanci)
        extra_names = {f"espn:{n}": n for n in roster_names if key(n) not in index}
        pids = on_roster
        roster = on_roster
        returning = {p for p in on_roster if p not in set(cur["player_id"])}
    last_any = cur.groupby("player_id")["game_date"].max()
    recent_min = cur.sort_values("game_date").groupby("player_id").tail(bf.WINDOW) \
        .groupby("player_id")["min"].mean()

    # srednie z 5 ostatnich rozegranych meczow (ten albo poprzedni sezon, dowolna druzyna)
    hist = h.pg[(h.pg["game_date"] < ts) & h.pg["player_id"].isin(pids)]
    hist = hist.sort_values("game_date").groupby("player_id").tail(5).copy()
    for c in ("pts", "reb", "ast", "blk", "stl", "tov", "per_num", "plus_minus"):
        if c not in hist:
            hist[c] = np.nan
    hist["min_per"] = hist["min"].where(hist["per_num"].notna())
    last5 = hist.groupby("player_id").agg(
        pkt5=("pts", "mean"), zb5=("reb", "mean"), as5=("ast", "mean"), min5=("min", "mean"),
        blk5=("blk", "mean"), prz5=("stl", "mean"), str5=("tov", "mean"),
        pm5=("plus_minus", "mean"),
        per_sum=("per_num", "sum"), per_min=("min_per", "sum"), ost=("game_date", "max"))
    lg = h.lg_per.get(s) or h.lg_per.get(s - 1)
    last5["per5"] = np.where((last5["per_min"] > 0) & bool(lg),
                             15 * last5["per_sum"] / last5["per_min"].where(last5["per_min"] > 0) /
                             (lg or 1), np.nan)

    rows = []
    for pid in sorted(pids):
        in_win = pid in agg.index
        l5 = last5.loc[pid] if pid in last5.index else None
        avg_min = float(agg.loc[pid, "avg_min"]) if in_win else np.nan
        if in_win:
            exp, plays = avg_min, pid in roster
        elif roster_names is not None and pid in recent_min.index:
            # np. po wymianie: minuty z ostatnich meczow w poprzedniej druzynie; gra, jesli
            # gral w ciagu ostatnich 14 dni
            exp = float(recent_min[pid])
            plays = (ts - last_any[pid]).days <= 14
        else:
            exp, plays = h.mpg_prev.get(pid, bf.NEW_PLAYER_MIN), False
        rows.append({
            "player_id": pid,
            "zawodnik": h.names.get(pid, pid),
            "status": "",
            "gra": bool(plays),
            "w_druzynie": True,
            "min": round(min(exp, 40.0), 1),
            "ocena": h.ratings.get((s - 1, pid), bf.REPLACEMENT),
            "avg_min": avg_min,
            "avg_pts": float(agg.loc[pid, "avg_pts"]) if in_win else np.nan,
            "w_oknie": in_win,
            "w_skladzie": pid in roster,
            "uwagi": "" if in_win else ("wraca (brak meczow w sezonie)" if pid in returning
                                        else "nie gral w 10 ostatnich meczach"),
            **{c: (round(float(l5[c]), 1) if l5 is not None and pd.notna(l5[c]) else np.nan)
               for c in ("min5", "pkt5", "zb5", "as5", "blk5", "prz5", "str5", "per5", "pm5")},
            "ost_mecz": f"{l5['ost']:%d.%m.%Y}" if l5 is not None else "",
        })
    for pid, name in extra_names.items():
        rows.append({"player_id": pid, "zawodnik": name, "status": "", "gra": False,
                     "w_druzynie": True, "min": float(bf.NEW_PLAYER_MIN), "ocena": bf.REPLACEMENT,
                     "avg_min": np.nan, "avg_pts": np.nan, "w_oknie": False, "w_skladzie": True,
                     "uwagi": "brak meczow w danych NBA", "ost_mecz": "",
                     **{c: np.nan for c in ("min5", "pkt5", "zb5", "as5", "blk5", "prz5", "str5",
                                            "per5", "pm5")}})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    # zawodnik wymieniony do innej druzyny nie jest w skladzie
    if roster_names is None:
        t = t[t["w_skladzie"] | ~t["player_id"].isin(played_for_team)]
    return t.sort_values(["gra", "min"], ascending=[False, False]).reset_index(drop=True)


def lineup_features(t: pd.DataFrame) -> dict:
    """missing_*, lineup_rating, missing_rating z tabeli zawodnikow (z poprawkami uzytkownika).
    Zawodnik z odznaczonym 'w_druzynie' (np. wymieniony) jest pomijany calkowicie."""
    if "w_druzynie" in t:
        t = t[t["w_druzynie"].astype(bool)]
    if t.empty:
        return {k: 0.0 for k in ("missing_min", "missing_pts", "missing_core_min",
                                 "missing_core_n", "missing_rating")} | {"lineup_rating": np.nan}
    gra = t["gra"].astype(bool)
    absent = t["w_oknie"] & t["w_skladzie"] & ~gra & (t["avg_min"] >= bf.ROT_MIN)
    core = absent & (t["avg_min"] >= bf.CORE_MIN)
    playing = gra & (t["min"] >= bf.ROT_MIN)
    wsum = t.loc[playing, "min"].sum()
    return {
        "missing_min": t.loc[absent, "avg_min"].sum(),
        "missing_pts": t.loc[absent, "avg_pts"].sum(),
        "missing_core_min": t.loc[core, "avg_min"].sum(),
        "missing_core_n": int(core.sum()),
        "lineup_rating": 5 * (t.loc[playing, "min"] * t.loc[playing, "ocena"]).sum() / wsum
        if wsum > 0 else np.nan,
        "missing_rating": (t.loc[absent, "avg_min"] * t.loc[absent, "ocena"]).sum() / 48,
    }


def feature_row(feats: list, home: dict, away: dict, elo_row) -> pd.DataFrame:
    x = {}
    for f in feats:
        if f.startswith("diff_"):
            k = f[5:]
            if k == "elo":
                x[f] = elo_row["home_elo"] - elo_row["away_elo"]
            else:
                x[f] = home.get(k, np.nan) - away.get(k, np.nan)
        else:
            x[f] = elo_row[f]
    return pd.DataFrame([x])[feats]


# ---------------------------------------------------------------- injury report
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z ]", " ", s.lower()).split())


STATUS_RE = "|".join(st.lower() for st in STATUSES)


def squash(s: str) -> str:
    """Male litery, bez akcentow, spacji i znakow poza przecinkiem.
    PDF NBA po odczycie czesto nie ma spacji ('BostonCelticsBrown,JaylenOut'),
    wiec porownujemy teksty w tej postaci."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z,]", "", s)


def report_key(name: str) -> str:
    """'Jaren Jackson Jr.' -> 'jacksonjr,jaren' (format raportu: 'Nazwisko, Imie')."""
    parts = str(name).split()
    if len(parts) < 2:
        return ""
    return squash(" ".join(parts[1:])) + "," + squash(parts[0])


def parse_injury_pdf(file) -> str:
    import pdfplumber
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def match_injuries(text: str, players: dict, game_date: dt.date = None) -> tuple:
    """players: player_id -> nazwisko. Zwraca (player_id -> status, niedopasowane, daty meczow).
    Szuka w kazdej linii raportu 'nazwisko,imie' + status. Jesli podano game_date,
    bierze tylko linie meczow z tego dnia (raport bywa na kilka dni)."""
    keys = {report_key(n): pid for pid, n in players.items() if report_key(n)}
    found, missed = {}, []
    dates = sorted(set(re.findall(r"\d{2}/\d{2}/\d{4}", text)))
    want = game_date.strftime("%m/%d/%Y") if game_date else None
    if want not in dates:
        want = None          # raport z innego dnia: najnowsze statusy wszystkich zawodnikow
    current = None
    for line in text.splitlines():
        d = re.match(r"\s*(\d{2}/\d{2}/\d{4})", line)
        if d:
            current = d.group(1)
        if want and current and current != want:
            continue
        sq = squash(line)
        hit = False
        for key, pid in keys.items():
            m = re.search(re.escape(key) + "(" + STATUS_RE + ")", sq)
            if m:
                found[pid] = STATUSES[[x.lower() for x in STATUSES].index(m.group(1))]
                hit = True
        if not hit:
            m = re.search(r"([a-z]+,[a-z]+?)(" + STATUS_RE + ")", sq)
            if m:
                missed.append(f"{m.group(1)} ({m.group(2)})")
    return found, missed, dates


def has_players(text: str) -> bool:
    """Czy raport zawiera choc jednego zawodnika (a nie tylko 'NOT YET SUBMITTED')."""
    return any(re.search(r"[a-z]+,[a-z]+(" + STATUS_RE + ")", squash(line))
               for line in text.splitlines())


def fetch_nba_report(hours_back: int = 48) -> tuple:
    """Najnowszy oficjalny injury report NBA, w ktorym sa juz zawodnicy:
    (nazwa_pliku, bajty, nazwa_najnowszego_pliku) albo (None, None, najnowszy).
    Raporty, w ktorych wszystkie druzyny maja 'NOT YET SUBMITTED', sa pomijane.
    Nazwy plikow maja godzine publikacji (czas USA, ET) co 15 min."""
    import io
    newest = None
    t = dt.datetime.now(ZoneInfo("America/New_York")).replace(second=0, microsecond=0)
    t = t.replace(minute=t.minute - t.minute % 15)
    for _ in range(hours_back * 4):
        names = [t.strftime("%I_%M%p")] + ([t.strftime("%I%p")] if t.minute == 0 else [])
        for tm in names:
            url = NBA_INJURY_URL.format(d=t.strftime("%Y-%m-%d"), t=tm)
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                name = url.rsplit("/", 1)[1]
                newest = newest or name
                if has_players(parse_injury_pdf(io.BytesIO(r.content))):
                    return name, r.content, newest
        t -= dt.timedelta(minutes=15)
    return None, None, newest


# pelne nazwy druzyn (tak jak w raporcie NBA) - do rozpoznania "NOT YET SUBMITTED"
TEAM_NAMES = {
    "1610612737": "Atlanta Hawks", "1610612738": "Boston Celtics", "1610612751": "Brooklyn Nets",
    "1610612766": "Charlotte Hornets", "1610612741": "Chicago Bulls",
    "1610612739": "Cleveland Cavaliers", "1610612742": "Dallas Mavericks",
    "1610612743": "Denver Nuggets", "1610612765": "Detroit Pistons",
    "1610612744": "Golden State Warriors", "1610612745": "Houston Rockets",
    "1610612754": "Indiana Pacers", "1610612746": "LA Clippers", "1610612747": "Los Angeles Lakers",
    "1610612763": "Memphis Grizzlies", "1610612748": "Miami Heat", "1610612749": "Milwaukee Bucks",
    "1610612750": "Minnesota Timberwolves", "1610612740": "New Orleans Pelicans",
    "1610612752": "New York Knicks", "1610612760": "Oklahoma City Thunder",
    "1610612753": "Orlando Magic", "1610612755": "Philadelphia 76ers", "1610612756": "Phoenix Suns",
    "1610612757": "Portland Trail Blazers", "1610612758": "Sacramento Kings",
    "1610612759": "San Antonio Spurs", "1610612761": "Toronto Raptors", "1610612762": "Utah Jazz",
    "1610612764": "Washington Wizards",
}


def report_team_status(text: str, game_date: dt.date = None) -> dict:
    """team_id -> 'submitted' albo 'nys' (NOT YET SUBMITTED) wg raportu NBA.
    Druzyny, ktorych w raporcie nie ma, nie pojawiaja sie w wyniku."""
    dates = set(re.findall(r"\d{2}/\d{2}/\d{4}", text))
    want = game_date.strftime("%m/%d/%Y") if game_date else None
    if want not in dates:
        want = None
    names = {tid: squash(n) for tid, n in TEAM_NAMES.items()}
    out, current = {}, None
    for line in text.splitlines():
        d = re.match(r"\s*(\d{2}/\d{2}/\d{4})", line)
        if d:
            current = d.group(1)
        if want and current and current != want:
            continue
        sq = squash(line)
        for tid, n in names.items():
            if n in sq:
                out[tid] = "nys" if "notyetsubmitted" in sq else "submitted"
    return out


def fetch_latest_nba_report(hours_back: int = 24) -> tuple:
    """Najnowszy opublikowany injury report NBA (takze z 'NOT YET SUBMITTED'):
    (nazwa_pliku, bajty) albo (None, None)."""
    t = dt.datetime.now(ZoneInfo("America/New_York")).replace(second=0, microsecond=0)
    t = t.replace(minute=t.minute - t.minute % 15)
    for _ in range(hours_back * 4):
        names = [t.strftime("%I_%M%p")] + ([t.strftime("%I%p")] if t.minute == 0 else [])
        for tm in names:
            url = NBA_INJURY_URL.format(d=t.strftime("%Y-%m-%d"), t=tm)
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                return url.rsplit("/", 1)[1], r.content
        t -= dt.timedelta(minutes=15)
    return None, None


def report_time(name: str):
    """'Injury-Report_2026-11-03_05_45PM.pdf' -> czas publikacji (strefa ET)."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})_?(\d{2})?(AM|PM)", str(name))
    if not m:
        return None
    d, hh, mm, ap = m.groups()
    t = dt.datetime.strptime(f"{d} {hh}:{mm or '00'} {ap}", "%Y-%m-%d %I:%M %p")
    return t.replace(tzinfo=ZoneInfo("America/New_York"))


def fetch_espn_roster(abbr: str) -> list:
    """Aktualny sklad druzyny z ESPN (nazwiska). abbr: skrot ESPN, np. 'BOS', 'GS', 'UTAH'."""
    r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/"
                     f"{abbr.lower()}/roster", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return [a.get("displayName", "") for a in r.json().get("athletes", []) if a.get("displayName")]


def fetch_espn_injuries() -> list:
    r = requests.get(ESPN_INJURIES, headers=HEADERS, timeout=30)
    r.raise_for_status()
    out = []
    for team in r.json().get("injuries", []):
        for i in team.get("injuries", []):
            a = i.get("athlete", {})
            out.append({"name": a.get("displayName", ""), "status": i.get("status", ""),
                        "team": team.get("displayName", ""), "comment": i.get("shortComment", ""),
                        "date": i.get("date", "")})
    return out


def espn_status(status: str, comment: str) -> str:
    """ESPN czesto daje 'Day-To-Day', a dokladny status jest w komentarzu
    ('... is questionable for Tuesday's game', '... has been ruled out ...')."""
    st_ = str(status)
    if st_.lower() != "day-to-day":
        return st_
    c = str(comment).lower()
    for words, label in ((("ruled out", " out for", "will not play", "won't play"), "Out"),
                         (("doubtful",), "Doubtful"), (("questionable",), "Questionable"),
                         (("probable",), "Probable"),
                         (("available", "will play", "cleared to play"), "Available")):
        if any(w in c for w in words):
            return label
    return "Questionable"          # 'Day-To-Day' bez szczegolow = Questionable


def match_espn(rows: list, players: dict) -> tuple:
    """Dopasowanie listy ESPN do zawodnikow dzisiejszych druzyn (po nazwisku)."""
    index = {squash(n): pid for pid, n in players.items() if n}
    found, shown = {}, {}
    for r in rows:
        pid = index.get(squash(r["name"]))
        if pid:
            label = espn_status(r["status"], r.get("comment", ""))
            found[pid] = ESPN_STATUS.get(label.lower(), "Questionable")
            shown[pid] = label
    return found, shown


def apply_injuries(t: pd.DataFrame, inj: dict, questionable_plays: bool,
                   shown: dict = None) -> pd.DataFrame:
    plays = {"Out": False, "Doubtful": False, "Questionable": questionable_plays,
             "Probable": True, "Available": True}
    t = t.copy()
    for i, pid in enumerate(t["player_id"]):
        if pid in inj:
            t.loc[t.index[i], "status"] = (shown or {}).get(pid, inj[pid])
            if inj[pid] in ("Out", "Doubtful", "Questionable"):
                t.loc[t.index[i], "gra"] = plays[inj[pid]] and bool(t.loc[t.index[i], "gra"])
            else:                                  # Probable / Available - gra
                t.loc[t.index[i], "gra"] = True
    return t


def extra_player(h: History, pid: str, date: dt.date) -> dict:
    """Wiersz dla zawodnika dodanego recznie (np. po wymianie, jeszcze bez meczu w nowej druzynie)."""
    ts = pd.Timestamp(date)
    cur = h.pg[(h.pg["season"] == h.season) & (h.pg["game_date"] < ts) & (h.pg["player_id"] == pid)]
    recent = cur.sort_values("game_date").tail(bf.WINDOW)
    exp = recent["min"].mean() if len(recent) else h.mpg_prev.get(pid, bf.NEW_PLAYER_MIN)
    return {"player_id": pid, "zawodnik": h.names.get(pid, pid), "status": "", "gra": True,
            "w_druzynie": True,
            "min": round(min(float(exp), 40.0), 1),
            "ocena": h.ratings.get((h.season - 1, pid), bf.REPLACEMENT),
            "avg_min": np.nan, "avg_pts": np.nan, "w_oknie": False, "w_skladzie": False,
            "uwagi": "dodany recznie"}


# ---------------------------------------------------------------- kursy (The Odds API)
ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
# kolejnosc bukmacherow: pierwszy dostepny dla meczu (Betclic jest legalny w PL; francuskie
# serwisy maja podobne marze, wiec sa najblizszym przyblizeniem)
BOOK_PRIORITY = ["betclic_fr", "unibet_fr", "winamax_fr", "pmu_fr"]
BOOK_NAMES = {"betclic_fr": "Betclic (FR)", "unibet_fr": "Unibet (FR)", "winamax_fr": "Winamax (FR)",
              "pmu_fr": "PMU (FR)"}
ODDS_KEY_FILE = os.path.join(bf.DATA, "odds_api_key.txt")
ODDS_CACHE = os.path.join(bf.DATA, "odds_latest.json")
PL_TAX = 0.88                                    # w Polsce 12% podatku od stawki


def _team_from_name(name: str):
    names = {squash(n): t for t, n in TEAM_NAMES.items()}
    names[squash("Los Angeles Clippers")] = "1610612746"
    return names.get(squash(name))


def fetch_odds(api_key: str) -> tuple:
    """Kursy na zwyciezce (h2h) dla nadchodzacych meczow NBA: (DataFrame, pozostale kredyty)."""
    r = requests.get(ODDS_URL, params={"apiKey": api_key, "regions": "eu", "markets": "h2h",
                                       "oddsFormat": "decimal"}, timeout=30)
    r.raise_for_status()
    rows = []
    for ev in r.json():
        h, a = _team_from_name(ev.get("home_team")), _team_from_name(ev.get("away_team"))
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                price = {o.get("name"): o.get("price") for o in m.get("outcomes", [])}
                rows.append({"home_team_id": h, "away_team_id": a, "commence": ev.get("commence_time"),
                             "book": bk.get("key"), "home_price": price.get(ev.get("home_team")),
                             "away_price": price.get(ev.get("away_team")),
                             "last_update": m.get("last_update") or bk.get("last_update")})
    return pd.DataFrame(rows), r.headers.get("x-requests-remaining")


def odds_for_game(odds: pd.DataFrame, home_id: str, away_id: str, start_utc: str) -> dict:
    """{bukmacher: (kurs_gospodarz, kurs_gosc)} dla meczu (najblizszy termin, max 36 h roznicy)."""
    if odds is None or odds.empty:
        return {}
    d = odds[(odds["home_team_id"] == home_id) & (odds["away_team_id"] == away_id)].copy()
    if d.empty:
        return {}
    start = pd.to_datetime(start_utc, utc=True, errors="coerce")
    d["diff"] = (pd.to_datetime(d["commence"], utc=True, errors="coerce") - start).abs()
    d = d[d["diff"] <= pd.Timedelta(hours=36)]
    return {r.book: (r.home_price, r.away_price) for r in d.sort_values("diff").itertuples()
            if pd.notna(r.home_price) and pd.notna(r.away_price)}


def value(p_home: float, prices: dict) -> dict:
    """Kursy Betclic, szansa rynku bez marzy (Pinnacle, a gdy brak - Betclic) i value po podatku."""
    book = next((b for b in BOOK_PRIORITY if b in prices), None)
    bet = prices.get(book) if book else None
    ref = prices.get("pinnacle") or prices.get("betfair_ex_eu") or bet
    out = {"kurs_h": bet[0] if bet else np.nan, "kurs_a": bet[1] if bet else np.nan,
           "book": book, "rynek_h": np.nan, "ev_h": np.nan, "ev_a": np.nan}
    if ref:
        ih, ia = 1 / ref[0], 1 / ref[1]
        out["rynek_h"] = ih / (ih + ia)
    if bet:
        out["ev_h"] = p_home * bet[0] * PL_TAX - 1
        out["ev_a"] = (1 - p_home) * bet[1] * PL_TAX - 1
    return out


# ---------------------------------------------------------------- papierowe typy
def save_paper(rows: pd.DataFrame) -> int:
    rows = rows.copy()
    rows["zapisano"] = dt.datetime.now().isoformat(timespec="seconds")
    if os.path.exists(PAPER_CSV):
        old = pd.read_csv(PAPER_CSV, dtype={"espn_id": str})
        old = old[~old["espn_id"].isin(rows["espn_id"])]   # nowszy zapis zastepuje starszy
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(PAPER_CSV, index=False)
    return len(rows)


def evaluate_paper(features: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(PAPER_CSV):
        return pd.DataFrame()
    p = pd.read_csv(PAPER_CSV, dtype={"espn_id": str, "home_team_id": str, "away_team_id": str})
    p["game_date"] = pd.to_datetime(p["data"])
    res = features[["game_date", "home_team_id", "away_team_id", "margin"]]
    p = p.merge(res, on=["game_date", "home_team_id", "away_team_id"], how="left")
    p["wynik"] = p["margin"]
    p["trafiony"] = np.where(p["margin"].isna(), np.nan,
                             ((p["p_home"] > 0.5) == (p["margin"] > 0)).astype(float))
    p["blad_pkt"] = (p["pred_margin"] - p["margin"]).abs()
    p["zwyciezca"] = np.where(p["margin"].isna(), "", np.where(p["margin"] > 0, p["home"], p["away"]))
    if "moj_typ" not in p:
        p["moj_typ"] = np.nan
    p["moj_trafiony"] = np.where(p["margin"].isna() | p["moj_typ"].isna(), np.nan,
                                 (p["moj_typ"] == p["zwyciezca"]).astype(float))
    if "kurs" not in p:
        p["kurs"] = np.nan
    # zysk na 1 jednostke stawki, po 12% podatku: wygrana = 0.88 * kurs - 1, przegrana = -1
    p["zysk"] = np.where(p["moj_trafiony"].isna() | p["kurs"].isna(), np.nan,
                         np.where(p["moj_trafiony"] == 1, PL_TAX * p["kurs"] - 1, -1.0))
    return p.drop(columns=["margin", "game_date"])
