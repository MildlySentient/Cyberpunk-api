## 0. GitHub Workflow (Required)

- All edits, patches, or new files must be committed with a clear, descriptive message.
- Feature branches for major subsystem additions/overhauls; PR/merge only after review.
- Pull before every push.  
- No direct commits to main/master without review (if team).
- Use `.gitignore` for scratchpads, logs, local temp files.
- Tag major system releases (e.g., v1.0, v1.1-beta) in both repo and at top of `index.md`.
- Every new file must be referenced in its PR/commit message and linked in `index.md`.# Update Guidelines
# Update Guidelines
**Purpose:**  
Defines process for updating, patching, or expanding any core file, table, or subsystem in the Cyberpunk 2020 digital engine.  
Prevents file drift, redundancy, or lost references.

---

## 1. **Version Control**
- Increment file version in the header (or as a comment) after *any* change.
- For major revisions, note change summary and affected tables at top of file.

## 2. **File Naming**
- All filenames must be lowercase, underscores only, no spaces.
- New files must be listed in `index.md` and relevant overview .mds.
- Replace/rename only if *all* crosslinks are updated.

## 3. **Adding/Expanding Tables**
- Add new .tsv files for any new job type, faction, location, gear, or event category.
- Crosslink new files in:
  - `index.md`
  - Relevant overview file(s)
  - Any related generator or lookup table

## 4. **Removing/Deprecating Files**
- Only remove files if *all* references and links are first deleted or replaced.
- Deprecated files must be tagged as such in `index.md` until all dependencies are cleared.

## 5. **Data Integrity**
- All .tsv files: first row = header; fields must match header exactly.
- No duplicate rows for unique jobs, NPCs, or gear.
- All crosslinks must point to the *current* filename, not legacy.

## 6. **Testing/QA**
- After any update, test lookups and roll calls for new/changed files.
- Confirm no .md or .tsv references orphaned, no broken links.

## 7. **Documentation**
- Every new subsystem or file must have a one-line summary in its parent overview.
- Major changes (rules, structure) require a dated log entry at the top of this file.

---

## **Last Updated:**  
[Insert date and brief summary here after each edit]

---

*Failure to follow these guidelines will lead to file fragmentation, lookup errors, and broken automation. Treat this as non-optional system law.*