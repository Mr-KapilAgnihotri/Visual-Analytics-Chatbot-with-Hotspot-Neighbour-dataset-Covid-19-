"""
Step 1 of the ETL pipeline: clean the raw CSV.

What it fixes
-------------
1. Berlin appears as 12 separate districts (its boroughs, IDs 11001-11012) that
   all carry the name "Berlin". We give each its real borough name so that
   users (and the chatbot) can tell them apart.
2. District ID 16063 appears twice per day, once named "Wartburgkreis" and once
   "Eisenach", with IDENTICAL case figures (only the mobility columns differ):
   the case reporting already treated them as one unit (they merged officially
   in 2021). We keep one row per day, named "Wartburgkreis (incl. Eisenach)".
3. For 21 city / rural-district pairs the source SWAPPED the names: e.g. official
   district key 09162 is the city of Munich (1.49 million people, bordered by 3
   districts) but was labelled "München, Kreis", while 09184 (the rural district
   that encloses the city) was labelled "München, Landeshauptstadt". Evidence for
   every pair: official German district keys (AGS), populations, and the recovered
   borders (a city enclosed by its rural district borders only 1-3 districts).
   Baden-Württemberg, Darmstadt and Offenbach were already correct.
4. Column names are converted to short snake_case names that are easy to use
   in SQL and in Cypher.
5. The date column becomes a real DATE, the Hotspot flag becomes a boolean,
   and count columns become integers.

Run:  python -m etl.clean      (writes data/clean.parquet)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "hotspot_neighbors.csv"
CLEAN_PARQUET = ROOT / "data" / "clean.parquet"

BERLIN_BOROUGHS = {
    11001: "Berlin Mitte",
    11002: "Berlin Friedrichshain-Kreuzberg",
    11003: "Berlin Pankow",
    11004: "Berlin Charlottenburg-Wilmersdorf",
    11005: "Berlin Spandau",
    11006: "Berlin Steglitz-Zehlendorf",
    11007: "Berlin Tempelhof-Schöneberg",
    11008: "Berlin Neukölln",
    11009: "Berlin Treptow-Köpenick",
    11010: "Berlin Marzahn-Hellersdorf",
    11011: "Berlin Lichtenberg",
    11012: "Berlin Reinickendorf",
}

# (official city key, official rural-district key) whose names the source swapped
SWAPPED_NAME_PAIRS = [
    (3403, 3458),    # Oldenburg
    (3404, 3459),    # Osnabrück
    (6611, 6633),    # Kassel
    (7312, 7335),    # Kaiserslautern
    (9162, 9184),    # München
    (9163, 9187),    # Rosenheim
    (9261, 9274),    # Landshut
    (9262, 9275),    # Passau
    (9362, 9375),    # Regensburg
    (9461, 9471),    # Bamberg
    (9462, 9472),    # Bayreuth
    (9463, 9473),    # Coburg
    (9464, 9475),    # Hof
    (9561, 9571),    # Ansbach
    (9563, 9573),    # Fürth
    (9661, 9671),    # Aschaffenburg
    (9662, 9678),    # Schweinfurt
    (9663, 9679),    # Würzburg
    (9761, 9772),    # Augsburg
    (13003, 13072),  # Rostock
    (14713, 14729),  # Leipzig
]

RENAME = {
    "Reporting_Date": "date",
    "District_ID": "district_id",
    "District": "district",
    "Federal_State": "state",
    "Population": "population",
    "Longitude": "lon",
    "Latitude": "lat",
    "Total_Cases": "total_cases",
    "New_Cases": "new_cases",
    "7_Day_Cases": "cases_7d",
    "7_Day_Incidence": "incidence_7d",
    "Hotspot": "hotspot",
    "Number_of_Deaths": "deaths",
    "Daily_District_Mobility_Inward": "mobility_inward",
    "Value": "mobility_value",
    "Percentage": "mobility_pct",
    "Mobility_Change_Value": "mobility_change",
    "Neighbor_Total_Cases": "nb_total_cases",
    "Neighbor_New_Cases": "nb_new_cases",
    "Neighbor_7_Day_Cases": "nb_cases_7d",
    "Neighbor_7_Day_Incidence": "nb_incidence_7d",
    "Neighbor_Number_of_Deaths": "nb_deaths",
}

INT_COLS = ["population", "total_cases", "new_cases", "cases_7d", "deaths",
            "nb_total_cases", "nb_new_cases", "nb_cases_7d", "nb_deaths"]


EISENACH_TEMP_ID = 16056   # Eisenach's historic key; used only while rebuilding borders


def clean(raw_csv: Path = RAW_CSV, merge_eisenach: bool = True) -> pd.DataFrame:
    """merge_eisenach=False keeps Eisenach as a separate pseudo-district (ID 16056).
    The border reconstruction needs that, because the source added Eisenach AND
    Wartburgkreis separately into their neighbours' totals."""
    df = pd.read_csv(raw_csv).rename(columns=RENAME)

    # Fix 1: name the Berlin boroughs
    mask = df["district_id"].isin(BERLIN_BOROUGHS)
    df.loc[mask, "district"] = df.loc[mask, "district_id"].map(BERLIN_BOROUGHS)

    # Fix 2: Eisenach rows duplicate Wartburgkreis's case data -> keep one unit
    if merge_eisenach:
        df = df[df["district"] != "Eisenach"].copy()
        df.loc[df["district_id"] == 16063, "district"] = "Wartburgkreis (incl. Eisenach)"
    else:
        df.loc[df["district"] == "Eisenach", "district_id"] = EISENACH_TEMP_ID

    # Fix 3: swap back the names of city / rural-district pairs
    names = df.drop_duplicates("district_id").set_index("district_id")["district"]
    for city, rural in SWAPPED_NAME_PAIRS:
        a, b = names[city], names[rural]
        df.loc[df["district_id"] == city, "district"] = b
        df.loc[df["district_id"] == rural, "district"] = a

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["hotspot"] = df["hotspot"].astype(bool)
    for c in INT_COLS:
        df[c] = df[c].round().astype("int64")

    df = df.drop_duplicates(["district_id", "date"]).sort_values(["district_id", "date"])
    assert df["district_id"].nunique() == df["district"].nunique(), "district names must be unique"
    return df.reset_index(drop=True)


if __name__ == "__main__":
    out = clean()
    out.to_parquet(CLEAN_PARQUET, index=False)
    print(f"Clean data: {len(out):,} rows, {out['district_id'].nunique()} districts -> {CLEAN_PARQUET}")
