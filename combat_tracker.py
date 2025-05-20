# combat_tracker.py
# Modular Combat Tracker for Cyberpunk 2020 (Merged Logic)

from cyberpunk2020_engine import roll_d10

combatants = []
initiative_order = []
current_turn = 0  # Index in initiative_order (0-based)
current_round = 1 # Human-facing round number (starts at 1)

def add_combatant(name, ref, hp, type='NPC'):
    """Add a combatant with REF stat and base HP."""
    roll = roll_d10()
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
    """Generate and sort initiative order for this encounter."""
    global initiative_order, current_turn, current_round
    initiative_order = sorted(combatants, key=lambda x: x['init'], reverse=True)
    current_turn = 0
    current_round = 1

def get_current_actor():
    """Return current actor in initiative order."""
    if not initiative_order:
        return None
    return initiative_order[current_turn % len(initiative_order)]

def next_turn():
    """Advance turn order, update round/acted state."""
    global current_turn, current_round
    if not initiative_order:
        return None
    actor = initiative_order[current_turn % len(initiative_order)]
    actor['acted'] = True
    current_turn += 1
    # If we looped through all, reset 'acted' and increment round
    if current_turn % len(initiative_order) == 0:
        for c in initiative_order:
            c['acted'] = False
        current_round += 1
    return actor

def apply_wound(name, damage):
    """Apply damage to named combatant, update status."""
    for c in combatants:
        if c['name'] == name and c['status'] == 'alive':
            c['wounds'] += damage
            c['hp'] -= damage
            # Critical status (tweak thresholds for your house rules)
            if c['hp'] <= 0:
                c['status'] = 'dead'
            elif c['hp'] <= 2:
                c['status'] = 'mortally wounded'
            elif c['hp'] <= 4:
                c['status'] = 'unconscious'
            return c['status']
    return None

def remove_combatant(name):
    """Remove combatant from tracker."""
    global combatants, initiative_order
    combatants = [c for c in combatants if c['name'] != name]
    initiative_order = [c for c in initiative_order if c['name'] != name]

def list_combatants():
    """List all combatants and their states."""
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
    """Reset all combat data for a new encounter."""
    global combatants, initiative_order, current_turn, current_round
    combatants = []
    initiative_order = []
    current_turn = 0
    current_round = 1