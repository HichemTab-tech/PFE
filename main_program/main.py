#!/usr/bin/env python3
import argparse
import pandas as pd
from data_import import load_csv_data, load_devices
from data_prep import extract_time_params
from csa import run_csa

def build_price_profile():
    # Remplacez par vos tarifs réels si nécessaire
    return pd.Series(
        [0.10]*7 +   # 0-6: tarif bas
        [0.20]*4 +   # 7-10: pointe matinale
        [0.15]*6 +   # 11-16: tarif intermédiaire
        [0.22]*2 +   # 17-18: pointe soirée
        [0.10]*5,    # 19-23: tarif bas
        index=range(24)
    )

def build_load_profile(df_day, devices):
    # Charge restante = toutes les colonnes numÃ©riques sauf celles des appareils
    numeric = df_day.select_dtypes(include=['float64', 'int64']).columns
    smart   = [c for c in numeric for d in devices if d in c]
    baseline = df_day[numeric].drop(columns=smart).sum(axis=1)
    # Moyenne par heure
    lp = baseline.groupby(baseline.index.hour).mean().reindex(range(24), fill_value=0)
    return lp

def main():
    parser = argparse.ArgumentParser(
        description="Génère un planning CSA pour une date donnée ou pour l'été par défaut"
    )
    parser.add_argument(
        "--date", help="Date cible (YYYY-MM-DD)", default=None
    )
    args = parser.parse_args()

    # 1) Chargement des données et appareils
    df = load_csv_data("HomeC.csv")
    devices = load_devices("devices_with_w.json", "HomeC.csv")
    print(df.index.date)
    # 2) Filtrage par date ou saison
    if args.date:
        target = pd.to_datetime(args.date).date()
        df_day = df[df.index.date == target]
        if df_day.empty:
            print(f"Pas de données pour {target}")
            return
        df_slice = df_day
    else:
        # Par défaut, on travaille sur l'été
        df_slice = df[df['month'].isin(['June', 'July', 'August'])]

    # 3) Extraction des paramètres temporels
    params = extract_time_params(df_slice, devices)
    print("\nParamètres temporels:")
    for d, p in params.items():
        print(f"  {d}: α={p['alpha']//3600}h, β={p['beta']//3600}h, LOT={p['LOT']//60:.1f}min, m={p['m']:.1f}h")

    # 4) Construction des profils
    price_profile = build_price_profile()
    load_profile = build_load_profile(df_slice, devices)
    print("\nProfil prix (€/kWh):", price_profile.values)
    print("Profil charge baseload (kW):", load_profile.values)

    # 5) Lancement de CSA
    schedule = run_csa(params, price_profile, load_profile, seed=42)
    print("\n=== Planning optimal ===")
    for d, h in schedule.items():
        print(f"  {d}: {h:02d}:00")

if __name__ == "__main__":
    main()
