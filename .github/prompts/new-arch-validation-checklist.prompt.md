---
mode: ask
description: Run the new-architecture validation checklist on the current patch or changed files. Use after implementing a new op or fixing an accuracy regression before submitting a PR.
---

Run the **new-architecture validation checklist** defined in
`.github/agents-prototype/skills/new-arch-validation-checklist.md`.

1. Read the checklist file now.
2. For each item in the checklist, examine the current changes (`git diff HEAD` or the files
   listed in `agent-results/*/files_created`) and determine whether the item is `PASS`,
   `FAIL`, or `N/A`.
3. Report results in a markdown table with columns: `Category`, `Item`, `Status`, `Notes`.
4. For any `FAIL` item, provide a specific fix recommendation tied to the exact changed code.
5. Summarise: how many PASS / FAIL / N/A, and whether the patch is ready to submit.
