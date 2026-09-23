# OpenVINO Copilot Instructions

## Scope

These instructions apply to changes in this repository. Follow more specific
instructions and skills when they are available for the files being changed.
Skills are available through `.agents/skills/` (a symlink to `.claude/skills/`)
and should be loaded when their description matches the task. Component-specific
documentation and local README files take precedence over general assumptions.
For code reviews, start with `.agents/skills/code-review/SKILL.md` and then
load the component references selected by its path-routing table. Ownership
and automatic maintainer review requests are defined separately in
`.github/CODEOWNERS`.

## Repository practices

- Inspect the relevant implementation, tests, build files, and documentation
  before editing.
- Keep changes focused on the requested behavior. Preserve unrelated local
  work and do not rewrite or revert changes outside the task.
- Follow existing APIs, naming, ownership, error-handling, localization, and
  test patterns. Prefer existing helpers over duplicated logic.
- Preserve type safety and public API compatibility. Treat model data, files,
  and other external input as untrusted.
- Report errors explicitly using the repository's established mechanisms.
  Do not hide failures behind broad catches, silent defaults, or fallback
  behavior that changes semantics.
- Update directly related documentation and tests for user-visible or
  behavioral changes.

## Validation

- Run the smallest relevant formatter, linter, build, or test command after
  editing. Expand validation when targeted checks reveal broader impact.
- Do not claim validation passed when a required dependency, build, or test was
  unavailable; state the limitation and the exact command or failure.
- Review the final diff for accidental files, generated-file changes, secrets,
  debug output, and temporary artifacts.

## Safety

- Never expose credentials or secrets. Share private data or repository
  content with external services only within the user's authorized scope;
  this permits authorized GitHub pull requests and review comments.
- Do not commit secrets or create destructive changes without explicit
  authorization.
- Use the repository's dedicated skills for specialized tasks such as code
  review, frontend review, CI workflows, or frontend conversion work.

## Useful documentation

- [Contributor and developer documentation](../docs/dev/index.md)
- [Build guide](../docs/dev/build.md)
- [Coding style](../docs/dev/coding_style.md)
- [Testing and coverage](../docs/dev/test_coverage.md)
- [Debug capabilities](../docs/dev/debug_capabilities.md)
- [CI overview](../docs/dev/ci/github_actions/overview.md)
- [Smart CI](../docs/dev/ci/github_actions/smart_ci.md)
- [CI security](../docs/dev/ci/github_actions/security.md)
- [Core documentation](../src/core/README.md)
- [Frontend documentation](../src/frontends/README.md)
- [Plugin documentation](../src/plugins/README.md)
- [Python bindings](../src/bindings/python/README.md)
- [C bindings](../src/bindings/c/README.md)
- [JavaScript bindings](../src/bindings/js/README.md)

When working in a component, first look for its nearest `README.md`,
developer documentation, tests, and build instructions. For frontend work,
use the [frontend index](../src/frontends/README.md) to find the relevant
documentation and source directory.
