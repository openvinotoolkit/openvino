# OpenVINO Release Notes Editorial and Formatting Rules

## Purpose

These rules define the minimum editorial, evidence, and reStructuredText requirements for OpenVINO release notes. They are intended for component owners, release coordinators, writers, and reviewers.

This document may be supplied to an AI system together with
`CONTRIBUTING_DOCS.md` and `CONTRIBUTING_PR.md` to help a human review release
notes and publish them as a correct reStructuredText (`.rst`) file. The human
reviewer remains responsible for validating the proposed output, resolving
owner questions, and approving publication.

## Evidence standard

Every publishable statement must be traceable to authoritative evidence.

- Prefer a merged PR and its changed implementation or documentation files.
- Use tests, model-validation results, benchmark reports, package indexes, or hardware-enablement records when a statement depends on behavior rather than code presence.
- A commit title alone is supporting context, not sufficient proof for detailed performance, capacity, compatibility, or model-validation claims.
- Split compound bullets into independently verifiable claims. If one clause is unsupported, revise or remove that clause rather than approving the whole bullet.
- For work implemented in another repository, label it as requiring external verification. Absence from the main OpenVINO repository does not prove that an external claim is false.
- Do not infer final archive names, build IDs, package versions, benchmark values, dates, or supported hardware from patterns used by previous releases.
- Embargoed or potentially confidential material must have explicit publication clearance.
- `TBD` and unresolved reviewer questions are blocking unless an owner explicitly approves their publication or removal.

## Content selection

## Marketing-owned What's New section

- Never edit any content in the `What's New` section. This section is owned by
  Marketing, and only specifically authorized Marketing contributors may
  change it.
- Do not correct spelling, grammar, terminology, formatting, capitalization,
  product names, code names, links, performance claims, or factual content in
  this section, even when the change appears safe or is required elsewhere by
  this guide.
- When transferring release notes from an approved source document, reproduce
  the complete `What's New` section exactly as supplied.
- If a problem is found in `What's New`, record it and refer it to an
  authorized Marketing owner. Do not fix it in the release-note source.
- These restrictions override all other editorial rules in this document for
  content within `What's New`.

Include changes that materially affect users:

- New supported hardware, models, frameworks, operations, APIs, packages, or deployment paths.
- Meaningful performance, memory, accuracy, reliability, security, or compatibility improvements.
- Behavior changes, limitations, known issues, deprecations, removals, and migration requirements.
- Installation, packaging, supported-system, and dependency changes users must act on.

Normally exclude:

- Test-only work, CI maintenance, dependency automation, code cleanup, refactoring without user-visible effects, and internal diagnostics.
- Narrow bug fixes with no material user impact.
- Implementation details that do not help users understand an outcome or required action.
- Future plans unless they are needed to explain a current preview limitation or deprecation timeline.

When a technical change is important but too narrow for the release notes, document it in the relevant product article or API reference instead.

## Language and tone

- Use concise, factual, user-centered language.
- Describe the shipped outcome first; add implementation detail only when it clarifies impact or usage.
- Use past tense for completed changes: “Added,” “Enabled,” “Improved,” “Fixed,” “Removed”.
- Use present tense for current limitations or requirements: “Only bounded shapes are supported”.
- Moderate qualitative wording such as “significant” is acceptable when it is supported and useful to readers. Avoid unsupported superlatives or promotional wording such as “dramatically,” “seamless,” “production-ready,” or “best”.
- Do not claim performance, accuracy, memory savings, capacity, production readiness, or model validation without corresponding evidence.
- Avoid vague claims such as “latest security fixes,” “improved stability,” or “broader support” when the concrete change can be named.
- Preserve technically significant terminology, but define uncommon abbreviations on first use.
- Do not silently correct source meaning. Editorial corrections may fix grammar, spacing, spelling, or formatting only when they do not alter the claim; substantive changes require owner confirmation.
- Use parallel grammatical structure within a bullet list.
- Vary leading verbs within a section instead of repeating the same verb in consecutive bullets.
- End complete-sentence bullets with periods; short noun-only list entries may omit periods if used consistently.

## Product names and terminology

- Use `OpenVINO™` where the established page convention requires the trademarked product name; do not add the mark to code identifiers, package names, URLs, commands, or every repetition.
- Write `Physical AI` in prose. Preserve `PhysicalAI` only when it is an exact code identifier or externally fixed name.
- Preserve official capitalization for Intel®, OpenVINO, ONNX, PyTorch, TensorFlow Lite, WinGet, GitHub, Node.js, APIs, models, and hardware.
- Use the approved feature and model spelling consistently. For the 2026.3
  notes, confirmed editorial forms include `GenAI`, `EAGLE-3`, `Top-K`,
  `PagedAttention`, `YOLO26`, and `Qwen3-ASR`. Do not normalize identifiers or
  model names whose official spelling has not been confirmed.
- Use singular device names when identifying a device class: `CPU`, `GPU`, and
  `NPU`, rather than `CPUs`, `GPUs`, and `NPUs`.
- Use the complete approved product name, trademark, and branding, for example `Intel® Core™ Ultra processors Series 2`.
- Do not publish internal code names. Replace them with approved branded names, for example `BMG` with `Intel® Xe2 architecture` and `ARL-H` with the applicable `Intel® Core™ Ultra processor` name.
- Flag potentially unreleased Intel product names, projects, and internal or customer information for owner review before publication.
- Distinguish CPU, GPU, and NPU carefully. Do not generalize validation on one device to other devices.
- State preview or experimental status at the beginning of the relevant bullet.
- Use exact release versions when they are authoritative. Avoid abbreviations such as `26.2` in user-facing prose; use `2026.2`.

## APIs, code, commands, and identifiers

- Format APIs, classes, functions, properties, environment variables, endpoints, commands, flags, filenames, package names when shown as commands, and code values with RST inline literals: ````identifier````.
- Preserve the exact spelling and case of code identifiers.
- Check every API, command, function, property, and package name against an authoritative source for spelling and case; do not assume the source draft is correct.
- Do not use typographic quotation marks as a substitute for inline literals.
- Use literal blocks for multi-line commands rather than embedding long commands in prose.
- Explain user action when a new flag or property is required; do not list an internal property without context.

## Structure and RST formatting

Follow the repository heading convention from `CONTRIBUTING_DOCS.md`:

1. article title: `=`
2. release heading: `#`
3. major section: `+`
4. component subsection: `-`
5. lower subsection when necessary: `.`

Additional requirements:

- Make the underline at least as long as the heading.
- Keep source lines approximately 70–100 characters where practical.
- Use `*` for unordered lists and two-space indentation per nested level.
- Keep related sub-bullets directly under their parent; do not insert an unindented paragraph that breaks the list.
- Under `Previous 2026 releases`, use `.. dropdown::` and convert nested historical headings to bold or italic emphasis rather than RST section underlines.
- Do not place section headings inside a dropdown in a way that produces `Unexpected section title` errors.
- Use one `Known Issues` section and the established component/ID/description structure.
- Keep global deprecation and legal sections outside historical release dropdowns.
- Run `git diff --check` and a Sphinx build after structural edits.

## Links

- Use `:doc:` for internal documentation pages and relative source paths where possible.
- Use external RST links in the form `` `label <URL>`__ ``.
- Do not hard-code versioned internal HTML URLs when a Sphinx cross-reference is available.
- Link to the most specific authoritative source relevant to the reader.
- Validate external links and ensure archived material points to the correct release branch or archive.
- Do not add a link merely as evidence for reviewers when it is not useful to readers; reviewer evidence belongs in the evidence matrix.

## Model and hardware claims

- Identify the device scope explicitly: CPU, GPU, NPU, or a validated combination.
- Distinguish “conversion supported,” “compiles,” “runs,” “validated,” and “optimized”; these are not interchangeable.
- State precision, platform generation, model variant, and known limitation when they materially constrain the claim.
- A model name in a test, model builder, or documentation example does not by itself prove full supported-model status.
- Capacity claims such as model parameter size or minimum system memory require a reproducible validation record.

## Performance and quality claims

- Name the affected metric when known: time to first token, time per output token, throughput, memory usage, load time, accuracy, or compilation time.
- Prefer general, supportable descriptions such as “improved performance,” “reduced latency,” or “enhanced throughput” instead of specific performance values or percentages.
- Do not publish specific performance metrics, percentages, or numeric comparisons in release-note prose. Keep measurement details in the authoritative benchmark or validation report.
- Remove broad performance, accuracy, memory, reliability, and
  production-readiness implications when the evidence establishes only that a
  feature is supported. Preserve the supported feature itself.
- Qualify preview results and device-specific results clearly.

## Known issues

Each issue must include:

- Component.
- Stable issue ID when available.
- Affected configuration, device, model, or operation.
- Observable impact.
- Workaround or recommended alternative, when available.

Remove the empty issue entry if no known issues are approved for publication. Never publish `TBD` as a final known issue.

An empty `Known Issues` heading may remain temporarily during release
preparation when the responsible owner has explicitly confirmed that approved
issues will be supplied later. Do not add `TBD`, `N/A`, or invented issue
entries. Recheck the section before publication.

## Source feedback and review decisions

- Reconcile the latest source document with its comments, the accompanying
  reviewer checklist, and decisions recorded during review. A checklist or
  explicit reviewer decision may require omitting content that remains in the
  source document.
- Record intentional source-to-publication differences so a later source
  conversion does not restore rejected or embargoed content.
- When feedback moves content to a later release, remove it from the current
  notes instead of rephrasing it as a current feature.
- Remove empty component subsections rather than publishing `N/A`.
- Avoid duplicating removals in feature sections. Describe them once in
  `Deprecation and Support`, with the affected product or component named.
- Keep unresolved technical naming and version conflicts as owner questions;
  do not resolve them through grammar editing.

## Deprecation and discontinuation

- Use “deprecated” when a feature remains available but is scheduled for removal.
- Use “discontinued” or “removed” only when it is unavailable in the current release.
- State the replacement or migration path.
- Include a removal release only when the date or version is approved.
- Verify removals against the current branch, not only against a source document or previous release note.
- Do not carry forward old deprecation bullets automatically; reconcile them with current implementation and the approved release source.
- Move every deprecation, discontinuation, and removal out of feature sections and into the deprecation and support section.

## Review checklist

Before publication, confirm that:

- Every bullet has evidence or an explicit owner-confirmation requirement.
- External-component claims have been checked in their owning repositories.
- No claim is contradicted by a later revert.
- All model, hardware, performance, and capacity qualifiers are supported.
- There are no unresolved `TBD`, `N/A`, comments, or embargo questions.
- Terminology, identifiers, links, headings, lists, and dropdowns follow these rules.
- Changed prose contains `Physical AI`, not `PhysicalAI`, except for identifiers.
- Deprecations and removals match current behavior.
- The documentation builds successfully and the generated page is visually reviewed.

## Sources used to derive these rules

- `CONTRIBUTING_DOCS.md`: RST structure, heading levels, line length, and link conventions.
- `CONTRIBUTING_PR.md`: user-impact documentation requirement and release-branch policy.
- `docs/articles_en/about-openvino/release-notes-openvino.rst`: current release structure, component hierarchy, historical dropdown pattern, known issues, deprecations, and legal section.
- OpenVINO 2025 and earlier 2026 release notes: wording, organization, terminology, and established presentation patterns. Provide these previous releases as references during drafting and review.
- The 2026.3 DOCX conversion comments and the 2026.2/2026.3 branch evidence audit.
- OpenVINO marketing department internal knowledge transfer and training
  materials on release notes.

Where these sources do not define editorial language or evidence standards explicitly, this document records the conservative publication rules used for the 2026.3 audit.
