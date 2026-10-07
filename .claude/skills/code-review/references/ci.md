# CI, Build, and Dependencies Review

For regular GitHub Actions workflow, action, or CI-script changes, consult
[ov-github-actions-ci](../../ov-github-actions-ci/SKILL.md) for repository
conventions and validation guidance. Apply it to reviewing the proposed
changes; do not treat its implementation steps as instructions to modify code.

- Check workflow permissions, untrusted-input handling, path filters, Smart CI
  behavior, cache scope, runner requirements, and runtime cost.
- Check that reusable workflows and composite actions preserve their input and
  output contracts.
- For dependency changes, check licensing, provenance, compatibility, and the
  [dependency-review configuration](../../../../.github/dependency_review.yml).
- Check build, test, release, and commit-policy workflows when their behavior
  is affected.
- Prefer pinned, reproducible actions and explicit failure handling.
