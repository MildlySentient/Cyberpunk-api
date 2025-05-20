""" 
CYBERPUNK 2020 ROLL ENGINE
---------------------------
All rolls in the system must route through this file.
Inline logic, simulated rolls, or randoms outside this script are prohibited.

Functions Provided:
- roll_d10(): Core resolution mechanic (used for skill checks, initiative, etc.)
- roll_d6(): Common for damage, humanity loss, or hit location
- roll_d100(): Optional percentile systems or rare loot triggers

Usage:
  from cyberpunk2020_engine import roll_d10
  result = roll_d10() + skill + stat + modifier

""" 

import random

def roll_d10():
    """Rolls 1d10. Used for skill checks, initiative, and task resolution."""
    return random.randint(1, 10)

def roll_d6():
    """Rolls 1d6. Used for damage, humanity loss, etc."""
    return random.randint(1, 6)

def roll_d100():
    """Rolls 1d100. Used for tables, loot rarity, or GM fiat."""
    return random.randint(1, 100)
    
    import random


