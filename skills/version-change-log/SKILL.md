---
name: version-change-log
description: Use when a release candidate (rc) branch is being released and CHANGELOG.md needs updating, or when asked to roll the changelog forward for a new version.
---

# Version Change Log

Roll `CHANGELOG.md` forward when an rc ships. Keep a Changelog format.

## Steps

1. `git fetch origin`, then diff `CHANGELOG.md` between `origin/main` and the rc branch (`origin/<X.Y.Z>-rc`).
2. Entries present on `main` but not in the rc's section go into the new version section
   (`## [next] - Unreleased`), under the matching subsection.
3. In the rc version section, replace `Unreleased` with today's date (`YYYY-MM-DD`).
4. New version section gets all subsections: Added, Changed, Deprecated, Removed, Fixed, Security.
5. Delete any empty subsections from the rc version section.
6. Update the link refs at the bottom: `[next]` compares `v<rc>...HEAD`, `[rc]` compares `v<prev>...v<rc>`.

## Notes

- Never remove empty subsections from the new (Unreleased) section.
- Don't edit entries' wording; only move them.
