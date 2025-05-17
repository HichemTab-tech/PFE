

equi = ["House overall [kW]", "Dishwasher [kW]", "Furnace 1 [kW]", "Furnace 2 [kW]", "Home office [kW]", "Fridge [kW]", "Wine cellar [kW]", "Garage door [kW]", "Kitchen 12 [kW]", "Kitchen 14 [kW]", "Kitchen 38 [kW]", "Barn [kW]", "Well [kW]", "Microwave [kW]", "Living room [kW]", "Solar [kW]"]

devices = {
    "Dishwasher [kW]": {
        "alpha": 9,      # Heure de début possible (par exemple, 9h)
        "beta": 17,      # Heure limite pour terminer (par exemple, 17h)
        "LOT": 2,        # Durée d'opération en heures (ex: 2h)
        "power": 1.2     # Puissance moyenne en kW
    },
    "Microwave [kW]": {
        "alpha": 8,
        "beta": 22,
        "LOT": 0.25,     # Durée d'opération : environ 15 minutes
        "power": 0.8
    },
    "Kitchen 12 [kW]": {
        "alpha": 7,
        "beta": 20,
        "LOT": 1,        # Utilisation sur 1 heure
        "power": 1.5
    },
    "Kitchen 14 [kW]": {
        "alpha": 7,
        "beta": 20,
        "LOT": 1,
        "power": 1.5
    },
    "Kitchen 38 [kW]": {
        "alpha": 7,
        "beta": 20,
        "LOT": 1,
        "power": 1.5
    },
}

# Affichage du dictionnaire pour vérifier
print("Paramètres des équipements flexibles:")
for device, params in devices.items():
    print(f"{device} : {params}")