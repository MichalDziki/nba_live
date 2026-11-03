"""
app.py - typy NBA na wybrany dzien (interfejs w przegladarce).

Uruchom w folderze nba_live:
    python3 -m pip install streamlit
    python3 -m streamlit run app.py

Przed dniem meczowym kliknij w aplikacji "Odswiez dane", zeby forma, minuty i Elo
byly aktualne. Kontuzje pobieraja sie z ESPN przy starcie i przyciskiem.
"""
import datetime as dt
import hashlib
import html
import io
import json
import os
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

import live
from train_eval import FEATURES_CSV

st.set_page_config(page_title="NBA typy", page_icon="🏀", layout="wide")
PL = ZoneInfo("Europe/Warsaw")


def secret(name, default=None):
    """Ustawienie z Streamlit Secrets (wersja w chmurze); lokalnie zwykle brak."""
    try:
        return st.secrets.get(name, default)
    except Exception:  # noqa: BLE001 - brak pliku secrets.toml
        return default


# Wersja w chmurze (Streamlit Community Cloud): dane i kursy aktualizuja sie same co 12 h,
# znajomi tylko ogladaja, przyciski zuzywajace kredyty sa za haslem, bez zapisu typow
# (dysk serwera jest kasowany przy restarcie), publicznie tylko kursy Betclic.
CLOUD = str(secret("CLOUD_MODE", "")).lower() in ("1", "true", "yes")
REFRESH_HOURS = 12
if CLOUD:
    live.BOOK_PRIORITY = ["betclic_fr"]


@st.cache_resource
def _refresh_lock():
    import threading
    return threading.Lock()


def run_refresh() -> str:
    logs = []
    for script in ("get_data.py", "build_features.py"):
        r = subprocess.run([sys.executable, script], capture_output=True, text=True)
        logs.append(f"$ {script}\n" + (r.stdout + r.stderr)[-1500:])
    st.cache_resource.clear()
    st.cache_data.clear()
    return "\n\n".join(logs)


def fetch_and_save_odds(key: str):
    try:
        odds_df, left = live.fetch_odds(key)
        with open(live.ODDS_CACHE, "w") as f:
            json.dump({"fetched": dt.datetime.now(PL).isoformat(), "remaining": left,
                       "rows": odds_df.to_dict("records")}, f)
        st.session_state.pop("odds_err", None)
    except Exception as e:  # noqa: BLE001
        st.session_state["odds_err"] = f"{type(e).__name__}: {str(e)[:80]}"


def older_than(path: str, hours: float) -> bool:
    return not os.path.exists(path) or (dt.datetime.now().timestamp() - os.path.getmtime(path)) > hours * 3600


if CLOUD:
    # dane: przy pierwszym uruchomieniu serwera i potem co 12 h (pierwszy odwiedzajacy czeka 1-3 min)
    if older_than(FEATURES_CSV, REFRESH_HOURS):
        with _refresh_lock():
            if older_than(FEATURES_CSV, REFRESH_HOURS):
                with st.spinner("Aktualizuję dane NBA – przy pierwszym uruchomieniu to trwa 2-5 minut..."):
                    os.makedirs("data", exist_ok=True)
                    st.session_state["refresh_log"] = run_refresh()
    key = secret("ODDS_API_KEY", "")
    if key and older_than(live.ODDS_CACHE, REFRESH_HOURS):
        with _refresh_lock():
            if older_than(live.ODDS_CACHE, REFRESH_HOURS):
                fetch_and_save_odds(key)

SHOW = ["zawodnik", "status", "gra", "min5", "pkt5", "zb5", "as5", "blk5", "prz5", "str5", "per5"]
HIDE = ["player_id", "w_druzynie", "min", "ost_mecz", "pm5", "ocena", "uwagi", "avg_min", "avg_pts",
        "w_oknie", "w_skladzie"]


def stat_col(label, help_=None):
    return st.column_config.NumberColumn(label, disabled=True, format="%.1f", width=48,
                                         help=help_)


EDITOR_CONFIG = {
    "zawodnik": st.column_config.TextColumn("Zawodnik", disabled=True, width=150),
    "status": st.column_config.TextColumn("Status", disabled=True, width=90,
                                          help="Status z listy kontuzji ESPN"),
    "gra": st.column_config.CheckboxColumn("Gra", width=58,
                                           help="Odznacz, jeśli zawodnik nie zagra"),
    "min": st.column_config.NumberColumn("Prognoza min", min_value=0, max_value=48, step=1,
                                         width=115,
                                         help="Spodziewane minuty (średnia z 10 ostatnich meczów "
                                              "drużyny) – możesz zmienić, wpływają na siłę składu"),
    "min5": stat_col("Min", "Średnie z 5 ostatnich rozegranych meczów (Min ... PER)"),
    "pkt5": stat_col("Pkt"),
    "zb5": stat_col("Zb", "Zbiórki"),
    "as5": stat_col("As", "Asysty"),
    "blk5": stat_col("Blk", "Bloki"),
    "prz5": stat_col("Prz", "Przechwyty"),
    "str5": stat_col("Str", "Straty"),
    "per5": stat_col("PER", "Player Efficiency Rating (Hollinger) z 5 ostatnich meczów; "
                            "średnia ligowa = 15"),
    "ost_mecz": st.column_config.TextColumn("Ostatni mecz", disabled=True, width=95),
    **{c: None for c in HIDE},
}

TABLE_CSS = """
<style>
.nba-t {width:100%; border-collapse:collapse; font-size:0.92rem; margin:0.25rem 0 0.5rem}
.nba-t th {text-align:left; font-weight:600; padding:8px 10px; opacity:.75;
           border-bottom:1px solid rgba(128,128,128,.4)}
.nba-t td {padding:8px 10px; border-bottom:1px solid rgba(128,128,128,.18); vertical-align:top}
.nba-t th, .nba-t td {white-space:nowrap}
.nba-t .num {text-align:right; font-variant-numeric:tabular-nums}
.nba-t details summary {cursor:pointer; list-style:none}
.nba-t details summary::-webkit-details-marker {display:none}
.nba-t details[open] summary {opacity:.6}
.nba-t .inj {display:block; margin-top:3px; line-height:1.45}
.nba-t .st {opacity:.6; font-size:.85em}
</style>
"""


# ---------------------------------------------------------------- cache
def data_version() -> float:
    return os.path.getmtime(FEATURES_CSV) if os.path.exists(FEATURES_CSV) else 0.0


@st.cache_resource(show_spinner="Wczytuję dane...")
def get_history(season: int, version: float):
    return live.History(season)


@st.cache_resource(show_spinner="Uczę model...")
def get_model(season: int, version: float):
    return live.train_final(get_history(season, version).features, season)


@st.cache_data(ttl=600, show_spinner="Pobieram terminarz z ESPN...")
def get_schedule(date: dt.date) -> pd.DataFrame:
    fake = os.environ.get("NBA_SCHEDULE_CSV")      # tylko do testow bez internetu
    if fake:
        return pd.read_csv(fake, dtype=str) if date == dt.datetime.now(
            ZoneInfo("America/New_York")).date() else pd.DataFrame()
    return live.fetch_schedule(date)


@st.cache_data(ttl=3600, show_spinner=False)
def get_roster(abbr: str):
    """Aktualny sklad z ESPN; None, gdy sie nie uda (wtedy sklad zgadujemy z historii meczow)."""
    fake = os.environ.get("NBA_ROSTERS_JSON")          # tylko do testow bez internetu
    try:
        if fake:
            return pd.read_json(fake, typ="series").get(abbr)
        return live.fetch_espn_roster(abbr)
    except Exception:  # noqa: BLE001
        return None


def to_pl(utc: str):
    try:
        return dt.datetime.fromisoformat(str(utc).replace("Z", "+00:00")).astimezone(PL)
    except ValueError:
        return None


def local_time(utc: str) -> str:
    t = to_pl(utc)
    return t.strftime("%H:%M") if t else ""


def fetch_injuries():
    """Najnowszy raport NBA + lista ESPN. Druzyny z 'NOT YET SUBMITTED' biora dane z ESPN."""
    state = {"fetched": dt.datetime.now(PL)}
    try:
        fake_pdf = os.environ.get("NBA_REPORT_PDF")     # tylko do testow bez internetu
        name, raw = ((os.path.basename(fake_pdf), open(fake_pdf, "rb").read()) if fake_pdf
                     else live.fetch_latest_nba_report())
        if raw:
            state.update(nba_name=name, nba_text=live.parse_injury_pdf(io.BytesIO(raw)))
    except Exception:  # noqa: BLE001
        pass
    try:
        fake = os.environ.get("NBA_INJURIES_JSON")      # tylko do testow bez internetu
        state["espn_rows"] = (pd.read_json(fake).to_dict("records") if fake
                              else live.fetch_espn_injuries())
    except Exception:  # noqa: BLE001
        pass
    st.session_state["inj"] = state


def nba_label(name: str) -> str:
    """'Injury-Report_2026-11-03_06_30AM.pdf' -> '[6:30 a.m. ET](link)'."""
    t = live.report_time(name)
    url = "https://ak-static.cms.nba.com/referee/injury/" + name
    if not t:
        return f"[{name}]({url})"
    hh = t.hour % 12 or 12
    return f"[{hh}:{t:%M} {'a.m.' if t.hour < 12 else 'p.m.'} ET]({url})"


# ---------------------------------------------------------------- panel boczny
with st.sidebar:
    st.header("Ustawienia")
    us_today = dt.datetime.now(ZoneInfo("America/New_York")).date()
    date = st.date_input("Dzień meczowy (data w USA)", us_today)
    if st.button("Pobierz kontuzje", help="Najnowszy oficjalny injury report NBA "
                                          "(gdy niedostępny – lista ESPN)") or \
            "inj" not in st.session_state:
        with st.spinner("Szukam najnowszego injury reportu (do ok. 30 s)..."):
            fetch_injuries()
    q_plays = st.radio("Zawodnik 'Questionable'", ["gra", "nie gra"],
                       horizontal=True) == "gra"

    if CLOUD:
        pwd = st.text_input("Hasło administratora", type="password",
                            help="Tylko dla właściciela – odblokowuje pobieranie kursów i danych")
        is_admin = bool(pwd) and pwd == str(secret("ADMIN_PASSWORD", ""))
    else:
        is_admin = True
    api_key = secret("ODDS_API_KEY", "") or (open(live.ODDS_KEY_FILE).read().strip()
                                             if os.path.exists(live.ODDS_KEY_FILE) else "")
    if not is_admin:
        pass
    elif not api_key:
        k = st.text_input("Klucz The Odds API", type="password",
                          help="Darmowy klucz z the-odds-api.com – zapisze się w data/odds_api_key.txt")
        if k:
            with open(live.ODDS_KEY_FILE, "w") as f:
                f.write(k.strip())
            st.rerun()
    elif st.button("Pobierz kursy", help="Kursy bukmacherów z regionu EU (1 kredyt z 500 miesięcznie)"):
        with st.spinner("Pobieram kursy..."):
            fetch_and_save_odds(api_key)
    info_box = st.container()

    st.divider()
    if is_admin and st.button("Odśwież dane", help="Pobiera nowe wyniki i przelicza cechy (1-3 min)"):
        with st.spinner("Pobieram dane i liczę cechy..."):
            st.session_state["refresh_log"] = run_refresh()
        st.rerun()

st.title("🏀 NBA – typy na dzień")

season = live.season_of(date)
if not os.path.exists(FEATURES_CSV):
    st.error("Brak data/features.csv – uruchom get_data.py i build_features.py.")
    st.stop()
h = get_history(season, data_version())
lr, ridge, feats, n_train = get_model(season, data_version())


def load_odds():
    if not os.path.exists(live.ODDS_CACHE):
        return pd.DataFrame(), None
    d = json.load(open(live.ODDS_CACHE))
    df = pd.DataFrame(d["rows"], dtype=object)
    used, n = "brak", 0
    if "book" in df:
        for b in live.BOOK_PRIORITY:
            k = int((df["book"] == b).sum())
            if k:
                used, n = live.BOOK_NAMES[b], k
                break
    info = {"fetched": dt.datetime.fromisoformat(d["fetched"]), "remaining": d.get("remaining", "?"),
            "book": used, "n": n}
    return df, info


odds, odds_info = load_odds()


def next_game_text() -> str:
    now = dt.datetime.now(PL)
    for k in range(4):
        try:
            sched = get_schedule(us_today + dt.timedelta(days=k))
        except Exception:  # noqa: BLE001
            return "–"
        if sched.empty:
            continue
        starts = [t for t in (to_pl(u) for u in sched["start_utc"]) if t and t > now]
        if starts:
            first = min(starts)
            g = sched.iloc[[to_pl(u) == first for u in sched["start_utc"]].index(True)]
            return f"{first:%d.%m %H:%M} ({g['away']} @ {g['home']})"
    return "–"


with info_box:
    inj_state = st.session_state.get("inj", {})
    got = inj_state.get("fetched")
    st.caption(f"Następny mecz: {next_game_text()}")
    if inj_state.get("nba_name"):
        t = live.report_time(inj_state["nba_name"])
        local = f" ({t.astimezone(PL):%H:%M} u nas)" if t else ""
        st.caption(f"Kontuzje: raport NBA {nba_label(inj_state['nba_name'])} czasu USA{local} · "
                   f"pobrano {got:%H:%M}")
    elif inj_state.get("espn_rows") is not None:
        st.caption(f"Kontuzje: ESPN (raport NBA niedostępny) · pobrano {got:%H:%M}")
    else:
        st.caption("Kontuzje: nie udało się pobrać")
    if st.session_state.get("odds_err"):
        st.caption(f"Kursy: błąd pobierania ({st.session_state['odds_err']})")
    elif odds_info and odds_info["n"] == 0:
        st.caption(f"Kursy: brak kursów {', '.join(live.BOOK_NAMES[b] for b in live.BOOK_PRIORITY)} "
                   f"w API · pobrano {odds_info['fetched']:%d.%m %H:%M}")
    elif odds_info:
        st.caption(f"Kursy: {odds_info['book']} · pobrano {odds_info['fetched']:%d.%m %H:%M} · "
                   f"{odds_info['n']} meczów z kursem · zostało {odds_info['remaining']}/500 "
                   "kredytów")
    elif api_key and is_admin:
        st.caption("Kursy: jeszcze nie pobrane – kliknij 'Pobierz kursy'.")

with st.sidebar:
    with st.expander("Szczegóły danych"):
        st.caption(f"Ostatni mecz NBA: {h.last_nba:%Y-%m-%d} · łącznie z ESPN: "
                   f"{h.tg['game_date'].max():%Y-%m-%d} · model: {n_train} meczów")
        if h.gap_info:
            st.caption(h.gap_info)
        if h.espn_check is not None:
            st.caption(f"Kontrola ESPN vs NBA ({h.last_nba:%Y-%m-%d}):")
            st.dataframe(h.espn_check, hide_index=True)
        if h.gap_unmatched:
            st.caption("Zawodnicy z ESPN bez dopasowania: " + ", ".join(h.gap_unmatched))
        if "refresh_log" in st.session_state:
            st.code(st.session_state["refresh_log"])

last_game = h.tg["game_date"].max()
if not CLOUD and pd.notna(last_game) and (pd.Timestamp(date) - last_game).days > 2:
    st.warning(f"Dane kończą się na {last_game:%Y-%m-%d}. Kliknij 'Odśwież dane', "
               "żeby forma i minuty były aktualne.")

if CLOUD:
    tab_today, tab_paper = st.container(), None
else:
    tab_today, tab_paper = st.tabs(["Typy dnia", "Moje typy"])


# ---------------------------------------------------------------- tabela typow (HTML)
def injured_cell(t) -> str:
    """Pierwszy nieobecny zawodnik; jesli jest ich wiecej - '…' i rozwijana lista."""
    if t is None or t.empty:
        return "—"
    out = t[~t["gra"].astype(bool)]
    if out.empty:
        return "—"
    items = [f"{html.escape(r.zawodnik)} <span class='st'>{html.escape(r.status or 'ręcznie')}</span>"
             for r in out.itertuples()]
    if len(items) == 1:
        return items[0]
    first = html.escape(out.iloc[0]["zawodnik"])
    body = "".join(f"<span class='inj'>{i}</span>" for i in items)
    return f"<details><summary>{first} …</summary>{body}</details>"


def book_label(rows: list) -> str:
    """Nazwa bukmachera, z ktorego pochodzi wiekszosc kursow dnia."""
    books = [r.get("book") for r in rows if r.get("book")]
    if not books:
        return "bukmachera"
    top = max(set(books), key=books.count)
    return live.BOOK_NAMES.get(top, top)


def summary_html(rows: list) -> str:
    """Tabela typow; kursy Betclic (FR) w dwoch kolumnach pod wspolnym naglowkiem."""
    left = ["Godz.", "Mecz", "Gospodarz %", "Gość %", "Typ", "Różnica pkt"]
    right = ["Brakuje – gość", "Brakuje – gospodarz"]
    num = {"Gospodarz %", "Gość %", "Różnica pkt"}

    def th(c, extra=""):
        return f"<th class='{'num' if c in num else ''}' {extra}>{c}</th>"

    head1 = ("".join(th(c, "rowspan='2'") for c in left)
             + f"<th colspan='2' style='text-align:center'>Kurs {book_label(rows)}</th>"
             + "".join(th(c, "rowspan='2'") for c in right))
    head2 = "<th class='num'>Gospodarz</th><th class='num'>Gość</th>"

    def kurs(v):
        return f"{v:.2f}" if pd.notna(v) else "—"

    trs = []
    for r in rows:
        cells = [(r["Godz. (PL)"], ""), (r["Mecz"], ""), (f"{r['Gospodarz %']:.1f}", "num"),
                 (f"{r['Gość %']:.1f}", "num"), (f"<b>{r['Typ']}</b>", ""),
                 (f"{r['Różnica pkt (gosp.)']:+.1f}", "num"),
                 (kurs(r["kurs_h"]), "num"), (kurs(r["kurs_a"]), "num"),
                 (injured_cell(r["_inj_away"]), ""), (injured_cell(r["_inj_home"]), "")]
        trs.append("<tr>" + "".join(f"<td class='{c}'>{v}</td>" for v, c in cells) + "</tr>")
    return (TABLE_CSS + f"<table class='nba-t'><thead><tr>{head1}</tr><tr>{head2}</tr></thead>"
            f"<tbody>{''.join(trs)}</tbody></table>")


# ---------------------------------------------------------------- typy dnia
def render_today():
    try:
        games = get_schedule(date)
    except Exception as e:  # noqa: BLE001
        st.error(f"Nie udało się pobrać terminarza z ESPN: {e}")
        return
    if games.empty:
        st.info("Brak meczów NBA w tym dniu.")
        return

    ctx = live.team_context(h, games, date)
    elo = live.elo_for(h, games, date)

    tables = {}
    with st.spinner("Pobieram aktualne składy z ESPN..."):
        for g in games.itertuples():
            for team, abbr in ((g.home_team_id, g.home), (g.away_team_id, g.away)):
                tables[(g.espn_id, team)] = live.player_table(h, team, date, get_roster(abbr))

    players = {pid: n for t in tables.values() if not t.empty
               for pid, n in zip(t["player_id"], t["zawodnik"])}
    inj_key = "brak"
    state = st.session_state.get("inj", {})
    nba_inj, team_status = {}, {}
    if state.get("nba_text"):
        nba_inj, _, _ = live.match_injuries(state["nba_text"], players, date)
        team_status = live.report_team_status(state["nba_text"], date)
    espn_inj, espn_shown = (live.match_espn(state["espn_rows"], players)
                            if state.get("espn_rows") is not None else ({}, {}))
    if state.get("nba_text") or state.get("espn_rows") is not None:
        inj_key = hashlib.md5(str((sorted(nba_inj.items()), sorted(espn_inj.items()),
                                   sorted(team_status.items()))).encode()).hexdigest()[:8]
        for (gid, team), t in tables.items():
            if t.empty:
                continue
            if team_status.get(team) == "submitted":     # druzyna zglosila sie w raporcie NBA
                tables[(gid, team)] = live.apply_injuries(t, nba_inj, q_plays)
            else:                                         # NOT YET SUBMITTED / brak -> ESPN
                tables[(gid, team)] = live.apply_injuries(t, espn_inj, q_plays, espn_shown)
    else:
        st.warning("Nie udało się pobrać kontuzji – kliknij 'Pobierz kontuzje' albo "
                   "odznacz nieobecnych ręcznie w tabelach meczów.")

    summary_box = st.container()

    rows = []
    for g in games.itertuples():
        title = f"{g.away} @ {g.home}  ·  {local_time(g.start_utc)} (PL)"
        sides, inj_lists = {}, {}
        with st.expander(title):
            for team, abbr in ((g.away_team_id, g.away), (g.home_team_id, g.home)):
                st.markdown(f"**{abbr}** {'(gość)' if team == g.away_team_id else '(gospodarz)'}")
                t = tables[(g.espn_id, team)]
                if t.empty:
                    st.info("Brak danych o zawodnikach.")
                    edited = t
                else:
                    edited = st.data_editor(
                        t[SHOW + HIDE], column_config=EDITOR_CONFIG, hide_index=True,
                        width="stretch", key=f"ed_{date}_{g.espn_id}_{team}_{inj_key}")
                lf = live.lineup_features(edited)
                sides[team] = {**ctx[(f"LIVE_{g.espn_id}", team)], **lf}
                if not edited.empty:
                    # nie graja: kontuzjowani (i 'Questionable', jesli ustawisz 'nie gra')
                    # oraz zawodnicy odznaczeni recznie
                    lst = edited[~edited["gra"].astype(bool) &
                                 ((edited["status"] != "") | (edited["w_oknie"] & edited["w_skladzie"]))]
                    inj_lists[team] = lst.sort_values("min", ascending=False)
                st.caption(f"Siła składu: {lf['lineup_rating']:.1f} · Elo: "
                           f"{elo.loc[g.espn_id, 'home_elo' if team == g.home_team_id else 'away_elo']:.0f}")

        x = live.feature_row(feats, sides[g.home_team_id], sides[g.away_team_id], elo.loc[g.espn_id])
        p = float(lr.predict_proba(x)[0, 1])
        m = float(ridge.predict(x)[0])
        rows.append({
            "espn_id": g.espn_id, "data": date.isoformat(),
            "home_team_id": g.home_team_id, "away_team_id": g.away_team_id,
            "Godz. (PL)": local_time(g.start_utc), "Mecz": f"{g.away} @ {g.home}",
            "Gospodarz %": 100 * p, "Gość %": 100 * (1 - p),
            "Typ": g.home if p >= 0.5 else g.away, "Różnica pkt (gosp.)": m,
            "p_home": p, "pred_margin": m, "home": g.home, "away": g.away,
            "_inj_away": inj_lists.get(g.away_team_id), "_inj_home": inj_lists.get(g.home_team_id),
            **live.value(p, live.odds_for_game(odds, g.home_team_id, g.away_team_id, g.start_utc)),
        })

    with summary_box:
        st.markdown(summary_html(rows), unsafe_allow_html=True)
        st.caption("Różnica pkt > 0: przewaga gospodarza (typowy błąd ok. ±10,5 pkt). "
                   "Kliknij nazwisko z '…', żeby zobaczyć wszystkich nieobecnych. "
                   "Zmiany w tabelach zawodników od razu przeliczają typy.")

        def lines(t):
            if t is None or t.empty:
                return "<i>brak</i>"
            return "<br>".join(html.escape(n) for n in t["zawodnik"])

        total = sum(len(x) for r in rows for x in (r["_inj_away"], r["_inj_home"]) if x is not None)
        with st.expander(f"🩹 Kontuzjowani / nie grają ({total})"):
            for r in rows:
                if all(x is None or x.empty for x in (r["_inj_away"], r["_inj_home"])):
                    continue
                st.markdown(f"**{r['Mecz']}** · {r['Godz. (PL)']}")
                c1, c2 = st.columns(2)
                c1.markdown(f"<i>{r['away']} (gość)</i><br>" + lines(r["_inj_away"]),
                            unsafe_allow_html=True)
                c2.markdown(f"<i>{r['home']} (gospodarz)</i><br>" + lines(r["_inj_home"]),
                            unsafe_allow_html=True)
                st.divider()

    # ---------------------------------------------------------- twoje typy (jak na kuponie)
    st.divider()
    st.subheader("Twoje typy")
    st.caption(f"Wybierz zwycięzcę w meczach, które chcesz typować "
               f"(w nawiasie kurs {book_label(rows)}).")
    saved = {}
    if os.path.exists(live.PAPER_CSV):
        old = pd.read_csv(live.PAPER_CSV, dtype={"espn_id": str})
        if "moj_typ" in old:
            saved = dict(zip(old["espn_id"], old["moj_typ"]))
    picks = {}
    for r in rows:
        c1, c2 = st.columns([2, 3])
        c1.markdown(f"**{r['Mecz']}** · {r['Godz. (PL)']}")
        def lab(team, price):
            return f"{team} ({price:.2f})" if pd.notna(price) else f"{team} (—)"
        labels = {lab(r["away"], r["kurs_a"]): r["away"], lab(r["home"], r["kurs_h"]): r["home"]}
        prev = [k for k, v in labels.items() if v == saved.get(r["espn_id"])]
        with c2:
            choice = st.segmented_control("Zwycięzca", list(labels), key=f"pick_{date}_{r['espn_id']}",
                                          default=prev[0] if prev else None,
                                          label_visibility="collapsed")
        if choice:
            picks[r["espn_id"]] = labels[choice]

    # kurs laczny (zaklad na wszystkie wybrane mecze naraz)
    prices = []
    for r in rows:
        if r["espn_id"] in picks:
            prices.append(r["kurs_h"] if picks[r["espn_id"]] == r["home"] else r["kurs_a"])
    if prices:
        known = [p for p in prices if pd.notna(p)]
        total = float(np.prod(known)) if known else float("nan")
        miss = len(prices) - len(known)
        st.markdown(f"**Kurs łączny ({len(prices)} {'mecz' if len(prices) == 1 else 'mecze'}): "
                    + (f"{total:.2f}**" if known else "—**"))
        if known:
            st.caption(f"Wygrana z 10 zł po 12% podatku: {10 * live.PL_TAX * total:.2f} zł"
                       + (f" · bez kursu dla {miss} meczu – nie wliczony" if miss else ""))

    if CLOUD:
        st.caption("W wersji online typy nie są zapisywane – zapis działa w aplikacji na komputerze.")
    elif st.button("💾 Zapisz twoje typy", type="primary", disabled=not picks):
        res = pd.DataFrame([r for r in rows if r["espn_id"] in picks])
        res["moj_typ"] = res["espn_id"].map(picks)
        res["kurs"] = np.where(res["moj_typ"] == res["home"], res["kurs_h"], res["kurs_a"])
        keep = ["espn_id", "data", "home", "away", "home_team_id", "away_team_id",
                "p_home", "pred_margin", "Typ", "moj_typ", "kurs"]
        live.save_paper(res[keep].rename(columns={"Typ": "typ"}))
        st.success(f"Zapisano {len(picks)} typów na {date:%Y-%m-%d} – zobacz zakładkę 'Moje typy'.")


with tab_today:
    render_today()

# ---------------------------------------------------------------- moje typy
def render_paper():
    ev = live.evaluate_paper(h.results())
    if ev.empty or ev["moj_typ"].isna().all():
        st.info("Brak zapisanych typów. Wybierz zwycięzców w sekcji 'Twoje typy' na dole "
                "zakładki 'Typy dnia' i kliknij 'Zapisz twoje typy'.")
    else:
        ev = ev[ev["moj_typ"].notna()]
        done = ev[ev["moj_trafiony"].notna()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Typy z wynikiem", f"{len(done)} / {len(ev)}")
        c2.metric("Twoja trafność", f"{100 * done['moj_trafiony'].mean():.1f}%" if len(done) else "–")
        c3.metric("Model na tych meczach", f"{100 * done['trafiony'].mean():.1f}%" if len(done) else "–")
        z = done["zysk"].dropna() if "zysk" in done else pd.Series(dtype=float)
        c4.metric("Zysk (1 j./typ, po podatku)", f"{z.sum():+.2f} j." if len(z) else "–",
                  f"ROI {100 * z.mean():+.1f}%" if len(z) else None)
        st.caption("Wyniki pojawiają się po odświeżeniu danych. Zysk liczony z kursu z chwili "
                   "zapisu i 12% podatku od stawki.")
        show = ev[["data", "away", "home", "moj_typ", "kurs", "typ", "zwyciezca", "moj_trafiony",
                   "zysk"]].copy()
        show["moj_trafiony"] = show["moj_trafiony"].map({1.0: "✅", 0.0: "❌"}).fillna("")
        show["zysk"] = show["zysk"].round(2)
        st.dataframe(show.rename(columns={"data": "Data", "away": "Gość", "home": "Gospodarz",
                                          "moj_typ": "Twój typ", "typ": "Typ modelu",
                                          "zwyciezca": "Zwycięzca", "moj_trafiony": "",
                                          "kurs": "Kurs", "zysk": "Zysk"}),
                     hide_index=True, width="stretch")


if tab_paper is not None:
    with tab_paper:
        render_paper()
