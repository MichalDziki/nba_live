"""
get_data.py - pobiera dane NBA z publicznego zbioru (github.com/llimllib/nba_data)
przechowywanego pod basketball-data.billmill.org. Zrodlo odswieza sie co 4 godziny
i zawiera biezacy sezon, wiec nie potrzebujemy stats.nba.com.

Uruchom:
    python3 -m pip install pyarrow requests pandas
    python3 get_data.py

Wynik: folder data/nba/ z plikami <zbior>_<sezon>.parquet
  team_game_logs   - druzyna x mecz (tradycyjne + zaawansowane: off/def/net rating, pace, poss)
  player_game_logs - zawodnik x mecz (tylko ci, ktorzy zagrali; min = minuty)
  games            - lista meczow (typ meczu, gospodarz/gosc)
  team_seasons     - id druzyny -> skrot (np. BOS)
  player_seasons   - id zawodnika -> nazwisko
  openrapm         - ocena zawodnika w sezonie (pkt na 100 posiadan, + = dobry)

Sezon oznaczony rokiem koncowym: 2010 = 2009-10, ..., 2027 = 2026-27.
Zakonczone sezony sa pobierane raz; biezacy (2027) zawsze od nowa.
"""
import os
import sys

import requests

ROOT = "https://basketball-data.billmill.org/nba"
# nazwa zbioru -> folder na serwerze
DATASETS = {
    "team_game_logs": "stats",
    "player_game_logs": "stats",
    "games": "stats",
    "team_seasons": "stats",
    "player_seasons": "stats",
    "openrapm": "openrapm",      # oceny zawodnikow (RAPM), od 2010
}
FIRST_SEASON = 2010
CURRENT_SEASON = 2027
OUT_DIR = os.path.join("data", "nba")


def download(url: str, path: str) -> bool:
    try:
        r = requests.get(url, timeout=120)
    except Exception as e:  # noqa: BLE001
        print(f"  BLAD polaczenia ({type(e).__name__}): {url}")
        return False
    if r.status_code == 404:
        print(f"  brak pliku (404): {url}")
        return False
    if r.status_code != 200:
        print(f"  BLAD {r.status_code}: {url}")
        return False
    with open(path, "wb") as f:
        f.write(r.content)
    return True


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    failures = 0
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        print(f"== Sezon {season - 1}-{str(season)[-2:]} (season={season}) ==")
        for ds, folder in DATASETS.items():
            path = os.path.join(OUT_DIR, f"{ds}_{season}.parquet")
            if os.path.exists(path) and season != CURRENT_SEASON:
                continue
            url = f"{ROOT}/{folder}/{ds}/season={season}/data.parquet"
            ok = download(url, path)
            if ok:
                print(f"  {ds}: {os.path.getsize(path) / 1024:.0f} KB")
            else:
                failures += 1
    print(f"Gotowe. Bledy/braki: {failures}")
    if failures:
        print("Wklej mi komunikaty powyzej, jesli cos sie nie pobralo.")
        sys.exit(1)


if __name__ == "__main__":
    main()
