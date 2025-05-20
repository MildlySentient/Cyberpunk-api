# Session Bootloader (`Bootloader.md`)

> Entry point for all Cyberpunk 2020 sessions  
> Standardizes runtime setup, tool loading, and system enforcement

---

## EXECUTION ORDER

```python
# Required: Only this engine for all rolls
exec(open("/mnt/data/cyberpunk2020_engine.py").read())

# Optional: Load for initiative/combat management
exec(open("/mnt/data/combat_tracker.py").read())
exec(open("/mnt/data/combat_resolution.py").read())
```

---

### REQUIRED FILES

| File                   | Purpose                         |
|------------------------|---------------------------------|
| cyberpunk2020_engine.py| All dice logic; no inline/random logic allowed |
| player_creation_flow.md| Stat and skill generation       |
| combat_tracker.py      | Combat initiative and wound states|

---

### OPTIONAL FILES

| File                   | Purpose                         |
|------------------------|---------------------------------|
| combat_resolution.py   | Damage, armor, injury resolution|
| npc_directory.tsv      | NPC stats and descriptions      |
| npc_templates.tsv      | Narrative archetypes            |
| story_hooks.tsv        | Story seeds, persistent threads |
| encounters_table.tsv   | Urban/random events             |
| faction_reactions.tsv  | Faction response/escalation     |
| job_generator.tsv      | Role-specific job/mission prompts|
| gear_tables.tsv        | Armor, gear, cyberware, upgrades|
| weapons_table.tsv      | Weapons, ammo, fire modes       |
| skills_table.tsv       | Skill definitions and uses      |
| roles_overview.md      | Playable roles; links to all role files |
| downtime_table.tsv     | Downtime, recovery, crafting    |
| economy_table.tsv      | Pricing, services, lifestyle    |
| psych_loss_table.tsv   | Trauma/psych loss tracking      |
| cyberware_table.tsv    | Chrome and costs                |
| cyberware_glitch_table.tsv| Cyberware malfunction/overload |
| cyberdecks_table.tsv   | Deck statlines, expansion       |
| difficulty_table.tsv   | Task modifiers                  |
| vehicles_table.tsv     | Vehicles and stats              |
| faults_table.tsv       | Flaws, faults, pressure triggers|
| referee_prompt.tsv     | Tone, pacing, referee/AI guidance|
| update_guidelines.md   | System update checklist         |

---

## LOAD ORDER & FALLBACKS

!load_order:
  1: cyberpunk2020_engine.py
  2: player_creation_flow.md
  3: combat_tracker.py
  4: referee_prompt.tsv
  5: npc_directory.tsv
  6: weapons_table.tsv
  7: gear_tables.tsv
  8: skills_table.tsv
  9: story_hooks.tsv
  10: faction_reactions.tsv
  11: encounters_table.tsv
  12: economy_table.tsv

---

## SYSTEM ENFORCEMENTS
- All dice logic via roll_d10() from cyberpunk2020_engine.py
- No inline, simulated, or estimated rolls
- All generated NPCs, gear, jobs, and narrative elements must reference a system file
- Any file not listed here or in the index is ignored by runtime

---

Update this file and index.md whenever you add or rename core system files.
