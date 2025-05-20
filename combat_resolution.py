""" 
COMBAT RESOLUTION MODULE
-------------------------
Handles weapon damage, armor mitigation, and hit location using TSVs:
- weapons_table.tsv
- gear_tables.tsv
- combat_actions_table.tsv

Main functions:
- resolve_attack()
- roll_hit_location()
- calculate_damage()
- apply_injury()
""" 

import random
import pandas as pd

# Load TSV tables (must exist in working directory or mounted path)
weapons_df = pd.read_csv("/mnt/data/weapons_table.tsv", sep="\t")
armor_df = pd.read_csv("/mnt/data/gear_tables.tsv", sep="\t")

def roll_hit_location():
    roll = random.randint(1, 10)
    locations = {
        1: "head",
        2: "right arm",
        3: "left arm",
        4: "chest",
        5: "chest",
        6: "abdomen",
        7: "right leg",
        8: "left leg",
        9: "right leg",
        10: "left leg"
    }
    return locations.get(roll, "chest")

def get_weapon(weapon_name):
    match = weapons_df[weapons_df['Name'].str.lower() == weapon_name.lower()]
    return match.iloc[0] if not match.empty else None

def get_armor_sp(armor_name, location):
    match = armor_df[armor_df['Name'].str.lower() == armor_name.lower()]
    if match.empty:
        return 0
    sp_field = f"SP_{location.lower().replace(' ', '_')}"
    return match.iloc[0].get(sp_field, 0)

def calculate_damage(weapon, armor_sp):
    try:
        dice_count, dice_size = map(int, weapon["Damage"].lower().split("d"))
        dmg = sum(random.randint(1, dice_size) for _ in range(dice_count))
    except:
        dmg = 0
    reduced = max(dmg - armor_sp, 0)
    return {"raw": dmg, "after_armor": reduced}

def resolve_attack(attacker, target, weapon_name, armor_name):
    weapon = get_weapon(weapon_name)
    if weapon is None:
        return {"error": "Weapon not found"}
    
    location = roll_hit_location()
    armor_sp = get_armor_sp(armor_name, location)
    damage = calculate_damage(weapon, armor_sp)

    result = {
        "attacker": attacker,
        "target": target,
        "weapon": weapon_name,
        "hit_location": location,
        "armor_sp": armor_sp,
        "raw_damage": damage["raw"],
        "damage_after_armor": damage["after_armor"],
        "status": "hit" if damage["after_armor"] > 0 else "no effect"
    }
    return result
