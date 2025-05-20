# Job Generator Index

| File Name              | Job Type/Domain           | Description                                                                                         | Source Book/Notes                  |
|------------------------|---------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------|
| fixer_jobs.tsv         | Fixer Jobs                | Classic contracts, street hustles, info brokering, open-world gigs from Night City’s fixers.        | Core, Wildside, scenario books      |
| gang_jobs.tsv          | Gang Jobs                 | Turf wars, violence, protection rackets, intimidation, body lotto, and organized crime gigs.        | Core, Wildside, Night City SB       |
| cop_jobs.tsv           | Cop Jobs                  | NCPD missions, detective cases, vice, blackmail, street enforcement, and police corruption.         | Core, Protect & Serve, scenario     |
| corporate_jobs.tsv     | Corporate Jobs            | Espionage, sabotage, black ops, asset recovery, R&D theft for or against the corps.                 | Core, Corpbooks, Night City SB      |
| nomad_pack_jobs.tsv    | Nomad Pack/Tribe Jobs     | Convoy defense, smuggling, trade, sabotage, environmental jobs, border running, family feuds.       | Neo-Tribes, Home of the Brave       |
| ripperdoc_jobs.tsv     | Ripperdoc/Black Market    | Illegal surgery, cyberware theft, organ recovery, black market installs, body disposal, sabotage.   | Chrome Books, Wildside, scenario    |
| media_jobs.tsv         | Media Jobs                | Investigations, exposes, rumor-chasing, PR gigs, scandal, sabotage, and news-driven crime.          | Core, Media, scenario books         |
| street_jobs.tsv        | Street/Underground Jobs   | Hustles, odd jobs, local fixer gigs, scavenging, survival schemes, street-level work.               | Core, Wildside, Night City SB       |
| netrunner_jobs.tsv     | Netrunner/Virtual Jobs    | Data heists, net sabotage, ICEbusting, AI contracts, blackmail, cybercrimes, and digital warfare.   | Core, Netrunner’s Handbook, Virtual Front |
| solo_jobs.tsv          | Solo/Mercenary Jobs       | Assassination, protection, black ops, personal vendetta, merc contracts, heavy combat.              | Core, Solo of Fortune, scenario     |
| trauma_team_jobs.tsv   | Trauma Team/Medical Jobs  | Emergency rescue, extractions, black market ambulance, illegal medical contracts, surgery runs.     | Trauma Team, Core, scenario books   |
| scenario_jobs.tsv      | Scenario/Specialist Jobs  | One-shots, high-concept campaign events, specialist missions, edge-case or splatbook content.       | All splatbooks, custom scenarios    |

---

## How to Use

- **Populate each .tsv with:**  
  - Job Type, Source/Broker, Target, Location, Complication, Reward, (optionally, Suggested Roles/Factions)
- **Crosslink from:**  
  - [fixers_table.tsv], [gangs_table.tsv], [corpo_brokers_table.tsv], [media_brokers_table.tsv], [ripperdoc_contacts_table.tsv], etc.  
  - for modular job assignment, lookup, and AI generation.
- **Expand with new files/categories** as new books, scenarios, or homebrew content require.
- **Link to:**  
  - [night_city_locations.tsv], [event_randomizers.tsv], [relationship_matrix.tsv] for dynamic locations, complications, and social/faction interplay.
- **Stub every new table** with at least 3–5 jobs for system “minimum viable playability.”

---

## Related Files

| File Name              | Description                                  |
|------------------------|----------------------------------------------|
| fixers_table.tsv       | Night City fixers and quest brokers          |
| gangs_table.tsv        | Gang contacts and criminal brokers           |
| corpo_brokers_table.tsv| Corporate job brokers/handlers               |
| media_brokers_table.tsv| Media/influencer contacts and news brokers   |
| ripperdoc_contacts_table.tsv | Ripperdoc job brokers/surgeons         |
| other_brokers_table.tsv| Specialist or edge-case job sources          |
| night_city_locations.tsv| City locations for gigs and complications   |
| event_randomizers.tsv  | Twist/event seed tables for extra chaos      |
| relationship_matrix.tsv| Faction, role, and rep interconnections      |

---

*Add, expand, and crosslink as the system grows. This structure keeps every job type discoverable and system-agnostic.*