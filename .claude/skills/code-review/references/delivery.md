# Code Review Delivery Contract

This contract is mandatory for every review produced with the `code-review`
skill (`../SKILL.md`). It overrides any default report layout.

## Why

Review UIs display the last assistant message of the session as the result.
If reasoning, planning, or a re-summary is emitted after the report, the report
and its tables are hidden behind that later message.

## Rules

1. Finish all investigation (file reads, searches, tool calls) before writing
   the report. Do not interleave investigation with the report.
2. Emit the complete report as a single final message, in this order:
   1. Main assessment and reviewed baseline (base/head SHAs; state whether
      inspection was complete or partial).
   2. Defect findings in descending order of impact, each with exact location,
      triggering condition, failure mechanism and impact, evidence, and a
      minimal fix direction. If none: "No actionable defects found within the
      inspected scope."
   3. Validation gaps and questions, kept separate from confirmed defects.
   4. Optional: reviewed-files table (`| File | Description |`).
   5. Review summary table. This MUST be the last element of the report:

      | Area | Assessment | Evidence / scope | Next action |
      | --- | --- | --- | --- |

      Assessment values: **Confirmed defect**, **Reported failure**,
      **Needs verification**, **Design question**, **No issue found** (only
      within the inspected scope), **Not inspected**. Use "None" as the next
      action when nothing is needed. The table must agree with the findings.
3. After the report, end the turn immediately: no further reasoning, planning,
   re-summarizing, or tool calls. To revise, re-emit the full report as the
   final message.
4. For PR reviews, when write access is available, also publish the report as
   a single PR review comment and post each confirmed defect as an inline
   comment at its exact file and line, so findings remain visible regardless of
   chat rendering.
5. Do not include secrets or sensitive data in the report or comments.
