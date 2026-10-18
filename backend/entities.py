"""
ENTITY LINKING: finds districts, states, dates and numbers in the user's text
and maps them to database identifiers.

    "Compare Munich and Köln in November"  ->
        districts [9162 München, Landeshauptstadt; 5315 Köln], start 2020-11-01, end 2020-11-30

How it works
  1. Every district / state gets several ALIASES: its official name, its bare
     name without "Kreis"/"Stadt" suffixes, and English names (Munich, Bavaria...).
  2. All text is "folded": lower-case, umlauts replaced (ü -> u), punctuation removed,
     so "München", "Muenchen" and "munchen" all look the same.
  3. The question is scanned for the LONGEST alias first (so "Brandenburg an der
     Havel" wins over "Brandenburg"), then for near-misses with fuzzy matching
     (rapidfuzz) to tolerate typos such as "Dusseldorff".
This is deterministic, needs no LLM, and runs in well under a millisecond.
"""
from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta

from rapidfuzz import fuzz, process

from backend import config

FIRST = date.fromisoformat(config.FIRST_DATE)
LAST = date.fromisoformat(config.LAST_DATE)

STATE_ALIASES = {
    "Freistaat Bayern": ["bavaria", "bayern"],
    "Baden-Württemberg": ["baden wurttemberg", "baden wuerttemberg", "bw"],
    "Nordrhein-Westfalen": ["north rhine westphalia", "nrw", "north rhine-westphalia"],
    "Niedersachsen": ["lower saxony"],
    "Hessen": ["hesse"],
    "Rheinland-Pfalz": ["rhineland palatinate", "rhineland-palatinate"],
    "Sachsen": ["saxony"],
    "Sachsen-Anhalt": ["saxony anhalt", "saxony-anhalt"],
    "Thüringen": ["thuringia"],
    "Mecklenburg-Vorpommern": ["mecklenburg western pomerania", "mecklenburg-western pomerania", "mv"],
    "Schleswig-Holstein": [],
    "Saarland": [],
    "Brandenburg": [],
    "Berlin": [],
    "Hamburg": [],
    "Bremen": [],
}
# English / common names for districts (keys must match district names exactly)
DISTRICT_ALIASES = {
    "München, Landeshauptstadt": ["munich", "munchen", "muenchen", "munich city"],
    "Köln": ["cologne"],
    "Nürnberg": ["nuremberg"],
    "Kreisfreie Stadt Frankfurt am Main": ["frankfurt", "frankfurt am main"],
    "Hannover": ["hanover"],
    "Braunschweig": ["brunswick"],
    "Düsseldorf": ["dusseldorf", "duesseldorf"],
    "Leipzig, Stadt": ["leipzig"],
    "Kreisfreie Stadt Kassel": ["kassel"],
    "Stadt Kaiserslautern": ["kaiserslautern"],
    "Rostock, Hansestadt": ["rostock"],
    "Wartburgkreis (incl. Eisenach)": ["eisenach", "wartburgkreis"],
    "Landeshauptstadt Wiesbaden": ["wiesbaden"],
    "Regionalverband Saarbrücken": ["saarbrucken", "saarbruecken", "saarbrücken"],
}
SUFFIXES = r",\s*(kreis|stadt|landeshauptstadt|hansestadt)$"
PREFIXES = r"^(kreisfreie stadt|stadt|landkreis|landeshauptstadt|regionalverband)\s+"
# words that must never be fuzzy-matched to a place name
STOPWORDS = set("""
cases case new total daily weekly monthly incidence rate hotspot hotspots deaths death mobility
compare comparison between versus trend over time show which what where when how many most least
top highest lowest district districts state states federal neighbour neighbours neighbor neighbors
border borders bordering adjacent within hops hop path route map share percentage breakdown with
their and the for from into during week month year germany german total population correlation
cluster clusters surrounded isolated connected chart graph plot
""".split())
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9


def fold(text: str) -> str:
    t = text.lower().replace("ß", "ss")
    t = t.replace("ä", "a").replace("ö", "o").replace("ü", "u")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Entity:
    kind: str            # "district" | "state"
    id: int | str        # district_id or state name
    name: str
    matched: str         # the text that matched
    score: float = 100.0


@dataclass
class Extraction:
    districts: list = field(default_factory=list)      # list[Entity]
    states: list = field(default_factory=list)         # list[Entity]
    date: str | None = None
    start: str | None = None
    end: str | None = None
    numbers: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class EntityLinker:
    def __init__(self, districts_df):
        """districts_df: DataFrame with district_id, name, state."""
        self.alias: dict[str, list[tuple[str, int | str, str]]] = {}
        self.district_name = dict(zip(districts_df["district_id"], districts_df["name"]))
        self.district_state = dict(zip(districts_df["district_id"], districts_df["state"]))
        self.id_by_name = {v: k for k, v in self.district_name.items()}
        self.states = sorted(districts_df["state"].unique())

        for did, name in self.district_name.items():
            for a in self._variants(name) + [fold(x) for x in DISTRICT_ALIASES.get(name, [])]:
                self._add(a, "district", did, name)
        for st in self.states:
            for a in [fold(st)] + [fold(x) for x in STATE_ALIASES.get(st, [])]:
                self._add(a, "state", st, st)

        # a bare name shared by a city and its rural district ("augsburg") means the CITY,
        # unless the user says "kreis"/"county"/"rural"; resolved in _pick
        self.sorted_aliases = sorted(self.alias, key=len, reverse=True)

    @staticmethod
    def _variants(name: str) -> list[str]:
        n = name.lower()
        out = {fold(name)}
        bare = re.sub(PREFIXES, "", re.sub(SUFFIXES, "", n))
        bare = re.sub(r"\s*\(.*?\)", "", bare)           # "halle (saale)" -> "halle"
        out.add(fold(bare))
        if n.startswith("berlin "):
            out.add(fold(n[len("berlin "):]))            # "neukolln"
        return [o for o in out if len(o) >= 3]

    def _add(self, alias, kind, id_, name):
        lst = self.alias.setdefault(alias, [])
        if (kind, id_, name) not in lst:
            lst.append((kind, id_, name))

    def _pick(self, alias: str, text: str, start: int) -> list[tuple]:
        cands = self.alias[alias]
        if len(cands) == 1:
            return cands
        districts = [c for c in cands if c[0] == "district"]
        states = [c for c in cands if c[0] == "state"]
        # "Hamburg"/"Bremen"/"Berlin"-style names that are both a state and a district
        if states and districts:
            around = text[max(0, start - 25): start + len(alias) + 25]
            return states if re.search(r"\bstate\b|\bland\b|\bbundesland\b", around) else districts[:1]
        if states:
            return states[:1]
        # city vs rural district of the same bare name
        around = text[max(0, start - 20): start + len(alias) + 25]
        want_rural = re.search(r"\b(kreis|landkreis|county|rural|district of)\b", around)
        cities = [c for c in districts if re.search(r"stadt|landeshauptstadt|hansestadt", c[2].lower())]
        rural = [c for c in districts if c not in cities]
        if want_rural and rural:
            return rural[:1]
        return (cities or districts)[:1]

    # ------------------------------------------------------------------
    def extract(self, question: str) -> Extraction:
        text = fold(question)
        ex = Extraction()
        taken = [False] * len(text)
        found: list[tuple[int, tuple, str, float]] = []

        # 1) exact alias matches, longest first, whole words only
        for alias in self.sorted_aliases:
            for m in re.finditer(rf"\b{re.escape(alias)}\b", text):
                s, e = m.span()
                if any(taken[s:e]):
                    continue
                if alias in STOPWORDS:
                    continue
                for c in self._pick(alias, text, s):
                    found.append((s, c, alias, 100.0))
                for i in range(s, e):
                    taken[i] = True

        # 2) fuzzy matches for leftover words / word pairs (typos)
        words = [(m.start(), m.group()) for m in re.finditer(r"[a-z]+", text)]
        for n in (2, 1):
            for i in range(len(words) - n + 1):
                s = words[i][0]
                chunk = " ".join(w for _, w in words[i:i + n])
                e = s + len(chunk)
                if any(taken[s:e]) or len(chunk) < 5 or any(w in STOPWORDS for _, w in words[i:i + n]):
                    continue
                hit = process.extractOne(chunk, self.sorted_aliases, scorer=fuzz.ratio, score_cutoff=86)
                if hit:
                    alias, score, _ = hit
                    for c in self._pick(alias, text, s):
                        found.append((s, c, chunk, score))
                    for k in range(s, min(e, len(taken))):
                        taken[k] = True

        for _, (kind, id_, name), matched, score in sorted(found, key=lambda f: f[0]):
            ent = Entity(kind, id_, name, matched, score)
            bucket = ex.districts if kind == "district" else ex.states
            if all(b.id != id_ for b in bucket):
                bucket.append(ent)

        self._dates(question, ex)
        ex.numbers = [int(n) for n in re.findall(r"\b(\d{1,3})\b", self._strip_dates(question))]
        return ex

    # ------------------------------------------------------------------
    # dates
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(d: date) -> date:
        return min(max(d, FIRST), LAST)

    @staticmethod
    def _strip_dates(q: str) -> str:
        q = re.sub(r"\d{4}-\d{2}-\d{2}", " ", q)
        q = re.sub(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", " ", q)
        q = re.sub(r"\b20\d\d\b", " ", q)
        months = "|".join(sorted(MONTHS, key=len, reverse=True))
        q = re.sub(rf"\b\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?({months})\b", " ", q, flags=re.I)
        q = re.sub(rf"\b({months})\s+\d{{1,2}}(st|nd|rd|th)?\b", " ", q, flags=re.I)
        return q

    def _parse_points(self, q: str) -> list[tuple[int, date, str]]:
        """All explicit dates/months in the text as (position, date, kind)."""
        pts = []
        ql = q.lower()
        for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", ql):
            pts.append((m.start(), date(int(m[1]), int(m[2]), int(m[3])), "day"))
        for m in re.finditer(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b", ql):
            y = int(m[3]) + (2000 if len(m[3]) == 2 else 0)
            pts.append((m.start(), date(y, int(m[2]), int(m[1])), "day"))
        months = "|".join(sorted(MONTHS, key=len, reverse=True))
        taken = [False] * len(ql)
        for pat, order in ((rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({months})\b", "dm"),
                           (rf"\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", "md")):
            for m in re.finditer(pat, ql):
                d_, mo = (m[1], m[2]) if order == "dm" else (m[2], m[1])
                try:
                    pts.append((m.start(), date(2020, MONTHS[mo], int(d_)), "day"))
                    for i in range(*m.span()):
                        taken[i] = True
                except ValueError:
                    pass
        for m in re.finditer(rf"\b({months})\b", ql):
            if any(taken[m.start():m.end()]):
                continue
            # "may" is usually the verb; accept it only after a preposition ("in May")
            if m[1] == "may" and not re.search(r"\b(in|during|of|for|since|until|to|from|and|between|early|late|mid)\s+may$",
                                               ql[:m.end()]):
                continue
            pts.append((m.start(), date(2020, MONTHS[m[1]], 1), "month"))
        return sorted(pts, key=lambda p: p[0])

    def _dates(self, q: str, ex: Extraction) -> None:
        ql = q.lower()
        holidays = {r"christmas eve": "2020-12-24", r"christmas( day)?|xmas": "2020-12-25",
                    r"new year'?s eve|silvester": "2020-12-31", r"boxing day": "2020-12-26",
                    r"easter( sunday)?": "2020-04-12"}
        for pat, day in holidays.items():
            if re.search(rf"\b({pat})\b", ql):
                ex.date = day
                ex.notes.append(f"interpreted as {day}")
                return
        if re.search(r"first wave", ql):
            ex.start, ex.end = "2020-03-01", "2020-05-31"
            ex.notes.append("first wave = March-May 2020")
            return
        if re.search(r"second wave", ql):
            ex.start, ex.end = "2020-10-01", "2020-12-31"
            ex.notes.append("second wave = October-December 2020")
            return
        if re.search(r"\b(whole|entire|full)\s+year\b|\bin 2020\b|\ball year\b", ql):
            ex.start, ex.end = config.FIRST_DATE, config.LAST_DATE
        pts = self._parse_points(q)
        if not pts:
            m = re.search(r"\blast (\d+) (day|week)s?\b", ql)
            if m:
                days = int(m[1]) * (7 if m[2] == "week" else 1)
                ex.start, ex.end = (LAST - timedelta(days=days - 1)).isoformat(), LAST.isoformat()
            return

        def month_end(d: date) -> date:
            return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])

        if len(pts) >= 2 and re.search(r"\b(between|from|since)\b", ql):
            (_, a, ka), (_, b, kb) = pts[0], pts[1]
            b = month_end(b) if kb == "month" else b
            ex.start, ex.end = self._clamp(a).isoformat(), self._clamp(b).isoformat()
        elif pts[0][2] == "day":
            ex.date = self._clamp(pts[0][1]).isoformat()
        else:
            d = pts[0][1]
            ex.start, ex.end = self._clamp(d).isoformat(), self._clamp(month_end(d)).isoformat()
