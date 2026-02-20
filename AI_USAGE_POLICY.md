# AI Usage Policy

## Purpose

OpenVINO welcomes responsible use of AI tools in open-source collaboration.
This policy exists to protect maintainer time, keep review quality high, and
ensure long-term maintainability of the project.

The key principle is simple: **contributions are evaluated by quality,
accountability, and maintainability, not by tool choice alone**.

## Scope

This policy applies to:

- Pull requests and code changes
- Issues and discussions
- Review communication on GitHub

## Allowed AI Assistance

You may use AI tools to assist your work, including but not limited to:

- Brainstorming and research
- Explaining APIs, language features, and error messages
- Drafting small self-contained snippets
- Refactoring suggestions
- Test ideas and documentation editing

All AI-assisted output must still meet OpenVINO contribution standards.

## Contributor Responsibilities

If you use AI in any meaningful way, you must:

1. **Understand your submission end-to-end** and be ready to explain design and
   implementation decisions.
2. **Verify correctness yourself** (build, tests, behavior, and edge cases).
3. **Take full responsibility for every line submitted**, regardless of how it
   was drafted.
4. **Disclose significant AI assistance** in the PR description.

Suggested disclosure format:

```text
AI assistance used: <no | yes>
If yes: <how AI was used>
Human validation performed: <build/tests/manual checks>
```

## Not Acceptable

The following are not acceptable and may lead to immediate closure of the
contribution:

- Submitting code you cannot explain or maintain
- Large, low-context, or low-quality AI-generated changes without thorough
  human validation
- Using AI-generated responses in place of direct, human-to-human communication
  during review
- Auto-generated issues/discussions that do not describe reproducible,
  first-hand observations
- Fabricated citations, benchmarks, bug reports, or security claims

## Review and Enforcement

Maintainers may:

- Ask for a clear explanation of any part of a contribution
- Request changes, reduction of scope, or additional tests
- Deprioritize or decline review when quality or ownership is unclear
- Close contributions that do not follow this policy or project guidelines

This policy is not based on AI detection. Enforcement is based on observed
contribution quality, reviewer confidence, and adherence to project rules.

## Practical Guidance for New Contributors

- Start with small, focused PRs
- Link each PR to a concrete issue when possible
- Avoid broad multi-component changes in a first contribution
- Prefer clear commit history and explicit rationale in PR descriptions

## Relationship to Other Project Policies

This document supplements, and does not replace:

- [Contributing Guidelines](./CONTRIBUTING.md)
- [PR Guidelines](./CONTRIBUTING_PR.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)

In case of conflict, maintainers' review decisions and repository governance
policies take precedence.