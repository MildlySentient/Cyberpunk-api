# combat_tracker.py
# Modular Combat Tracker for Cyberpunk 2020 — with typo-tolerant TSV lookup

import os
import pandas as pd
import random
import difflib

# --- Data loading abstraction ---
DATA_DIR = os.path.dirname(__file__)

def find_tsv(target, cutoff=0.85):
    """Fuzzy-match a .tsv file in DATA_DIR. Returns full path or raises."""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.tsv')]
    matches = difflib.get_close_matches(target, files, n=1, cutoff=cutoff)
    if not matches:
        raise FileNotFoundError(f"No TSV resembling '{target}' found. Searched: {files}")
    return os.path.join(DATA_DIR, matches[0])

def load_table(target, sep="\t", cutoff=0.85):
    """Load a TSV table by fuzzy-matched name."""
    path = find_tsv(target, cutoff)
    return pd.read_csv(path, sep=sep)

# Load weapon and armor tables (names can be approximate, e.g. 'weapon' finds 'weapons_table.tsv')
weapons_df = load_table("weapons_table.tsv")
armor_df = load_table("gear_upgrade_table.tsv")  # Adjust this to match your actual armor TSV

# --- Combat state (in-memory) ---
combatants = []
initiative_order = []
current_turn = 0   # Index in initiative_order (0-based)
current_round = 1  # Human-facing round (starts at 1)

def add_combatant(name: str, ref: int, hp: int, type: str = 'NPC'):
    """Add a combatant with REF stat and base HP."""
    roll = random.randint(1, 10)
    initiative = roll + ref
    combatants.append({
        'name': name,
        'ref': ref,
        'roll': roll,
        'init': initiative,
        'hp': hp,
        'wounds': 0,
        'status': 'alive',         # alive | dead | unconscious | mortally wounded
        'acted': False,            # Has acted this round?
        'type': type               # PC or NPC
    })

def roll_initiative():
    """Sort combatants by initiative."""
    global initiative_order, current_turn, current_round
    initiative_order = sorted(combatants, key=lambda x: x['init'], reverse=True)
    current_turn = 0
    current_round = 1

def start_combat(combatant_list):
    """Start a new combat with a provided list of combatants."""
    reset_combat()
    for c in combatant_list:
        if isinstance(c, dict):
            add_combatant(
                c.get('name', 'Unknown'), 
                c.get('ref', 5), 
                c.get('hp', 10), 
                c.get('type', 'NPC')
            )
        else:
            add_combatant(*c)
    roll_initiative()
    return list_combatants()

def get_current_actor():
    """Return the current actor in initiative order."""
    if not initiative_order:
        return None
    return initiative_order[current_turn % len(initiative_order)]

def get_current_turn():
    """Return info about whose turn it is now."""
    actor = get_current_actor()
    return actor if actor else {}

def next_turn():
    """Advance to the next turn, update acted/round state."""
    global current_turn, current_round
    if not initiative_order:
        return None
    actor = initiative_order[current_turn % len(initiative_order)]
    actor['acted'] = True
    current_turn += 1
    if current_turn % len(initiative_order) == 0:
        for c in initiative_order:
            c['acted'] = False
        current_round += 1
    return actor

def apply_wound(name: str, damage: int):
    """Apply damage to a combatant and update status."""
    for c in combatants:
        if c['name'] == name and c['status'] == 'alive':
            c['wounds'] += damage
            c['hp'] -= damage
            if c['hp'] <= 0:
                c['status'] = 'dead'
            elif c['hp'] <= 2:
                c['status'] = 'mortally wounded'
            elif c['hp'] <= 4:
                c['status'] = 'unconscious'
            return c['status']
    return None

def remove_combatant(name: str):
    """Remove a combatant by name."""
    global combatants, initiative_order
    combatants = [c for c in combatants if c['name'] != name]
    initiative_order = [c for c in initiative_order if c['name'] != name]

def list_combatants():
    """List all combatants and their states (ordered)."""
    return [
        {
            'name': c['name'],
            'initiative': c['init'],
            'hp': c['hp'],
            'wounds': c['wounds'],
            'status': c['status'],
            'acted': c['acted'],
            'type': c['type']
        }
        for c in initiative_order
    ]

def reset_combat():
    """Reset all combat state (use to start a new encounter)."""
    global combatants, initiative_order, current_turn, current_round
    combatants = []
    initiative_order = []
    current_turn = 0
    current_round = 1

# --- Combat Resolution Section ---
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

def conservative_name_match(df, col, name, cutoff=0.85):
    """Return row with the closest name match in a DataFrame column (case-insensitive, cutoff)."""
    names = df[col].astype(str).str.lower().tolist()
    matches = difflib.get_close_matches(name.lower(), names, n=1, cutoff=cutoff)
    if not matches:
        return None
    return df[df[col].str.lower() == matches[0]].iloc[0]

def get_weapon(weapon_name):
    return conservative_name_match(weapons_df, 'Name', weapon_name)

def get_armor_sp(armor_name, location):
    row = conservative_name_match(armor_df, 'Name', armor_name)
    if row is None:
        return 0
    sp_field = f"SP_{location.lower().replace(' ', '_')}"
    return row.get(sp_field, 0) if sp_field in row else 0

def calculate_damage(weapon, armor_sp):
    try:
        # Expects Damage like '2d6+1' or '1d10'
        import re
        m = re.match(r"(\d+)d(\d+)(\+\d+)?", str(weapon["Damage"]))
        dice_count = int(m.group(1))
        dice_size = int(m.group(2))
        bonus = int(m.group(3) or 0)
        dmg = sum(random.randint(1, dice_size) for _ in range(dice_count)) + bonus
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