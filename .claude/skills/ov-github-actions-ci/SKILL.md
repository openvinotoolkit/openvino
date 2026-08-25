---
name: ov-github-actions-ci
description: Author, edit, debug, and review the OpenVINO GitHub Actions CI infrastructure — regular (non-agentic) workflows under .github/workflows, reusable job_*.yml workflows, custom composite actions under .github/actions, CI helper scripts under .github/scripts, Dockerfiles under .github/dockerfiles, and the Smart CI / labeler / components configuration. Use when a user wants to add or change a build/test job or step, create or modify a reusable workflow, write or fix a custom action, adjust runners/containers/caches, wire up Smart CI for a component, pin action versions, fix workflow permissions/security, or debug a failing CI workflow's YAML. Do NOT use for gh-aw agentic workflows (*.md with gh-aw frontmatter — use ov-agentic-workflows), for diagnosing a specific product/test failure's root cause in C++/Python code, or for non-CI GitHub configuration.
---

# Work on OpenVINO GitHub Actions CI

Guides changes to the **regular** GitHub Actions CI in this repository: validation and reusable
workflows, custom actions, CI scripts, Dockerfiles, and the Smart CI configuration that drives them.

**Read the CI developer docs under [docs/dev/ci/github_actions](../../../docs/dev/ci/github_actions)
first** — they are the authoritative, repo-specific reference and this skill is a checklist on top of
them. Keep them in sync when behavior changes (see **Skill self-improvement**). The key pages:

| Topic | Doc |
|-------|-----|
| Big picture, workflow structure, triggers, results/artifacts/logs | [overview.md](../../../docs/dev/ci/github_actions/overview.md) |
| Reusable `job_*.yml` workflows | [reusable_workflows.md](../../../docs/dev/ci/github_actions/reusable_workflows.md) |
| Custom composite actions | [custom_actions.md](../../../docs/dev/ci/github_actions/custom_actions.md) |
| Smart CI (skip unaffected jobs) | [smart_ci.md](../../../docs/dev/ci/github_actions/smart_ci.md) |
| Runners (`runs-on`) | [runners.md](../../../docs/dev/ci/github_actions/runners.md) |
| Docker images / `handle_docker` | [docker_images.md](../../../docs/dev/ci/github_actions/docker_images.md) |
| Caches (GHA / shared drive / sccache) | [caches.md](../../../docs/dev/ci/github_actions/caches.md) |
| Adding tests (step / job / workflow) | [adding_tests.md](../../../docs/dev/ci/github_actions/adding_tests.md) |
| OpenVINO Provider (prebuilt artifacts) | [openvino_provider.md](../../../docs/dev/ci/github_actions/openvino_provider.md) |
| Workflow security | [security.md](../../../docs/dev/ci/github_actions/security.md) |

For framework syntax, use the official [GitHub Actions documentation](https://docs.github.com/en/actions). If user requires an out-of-scope feature, consult the official documentation.

**Out of scope:** `gh-aw` agentic workflows (`.github/workflows/*.md` + `*.lock.yml`, e.g. `ci-doctor`)
— use the **ov-agentic-workflows** skill instead.

## Repository layout

* **Workflows** — `.github/workflows/`
  * **Validation workflows**: named after the OS/config, e.g. `ubuntu_22.yml`,
    `windows_vs2022_release.yml`, `mac_arm64.yml`, `linux_arm64.yml`, `android.yml`. Entry points with
    `on:` triggers; they wire together `Build` + test jobs.
  * **Reusable workflows**: `job_*.yml` (e.g. `job_python_unit_tests.yml`, `job_cxx_unit_tests.yml`).
    Called via `uses: ./.github/workflows/job_*.yml` with `on: workflow_call:` inputs. **Not** triggered
    directly.
* **Custom actions** — `.github/actions/` (composite `action.yml`): `setup_python`, `system_info`,
  `smart-ci`, `handle_docker`, `openvino_provider`, `store_artifacts`/`restore_artifacts`, `cache`, etc.
* **CI scripts** — `.github/scripts/` (Python helpers: `workflow_rerun/`, `external_pr_labeller.py`,
  `check_copyright.py`, ...).
* **Dockerfiles** — `.github/dockerfiles/ov_build/**` and `ov_test/**`, plus the `docker_tag` file.
* **Smart CI config** — `.github/labeler.yml` (path globs → component/label) and
  `.github/components.yml` (component dependency graph).
* **Scans/meta** — `workflows_scans.yml` (CodeQL `actions` + semgrep on workflow changes),
  `dependency_review.yml`.

## Golden rules

1. **Pin third-party actions to a full commit SHA**, with the version in a trailing comment. First-party
   `./.github/actions/*` are referenced by path, not pinned.
   ```yaml
   uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
   ```
2. **Least-privilege `permissions:`.** Start from `permissions: read-all` (workflow level) and grant the
   minimum extra scope at the **job** level only where needed. Never widen without a reason.
3. **Follow [security.md](../../../docs/dev/ci/github_actions/security.md)** — it is the source of truth.
   Key rules: never use `pull_request_target`, never hard-code secrets, and treat all `github.*` /
   `github.event.*` user-controlled values as untrusted (route them through `env:` or action inputs, never
   interpolate directly into `run:` shell). Ping the CI task force for anything involving secrets or
   elevated triggers.
4. **Put shared job logic in a reusable `job_*.yml`, not copy-paste.** If the same job appears in more
   than one validation workflow, it should be a `workflow_call` reusable workflow parameterized by
   `runner`, `image`/`container`, and `affected-components`.
5. **Respect Smart CI.** Test jobs/steps that validate a specific component must gate on
   `fromJSON(needs.smart_ci.outputs.affected_components).<COMPONENT>.{build,test}` and list `Smart_CI` in
   `needs`. Do not make an expensive job run unconditionally.
6. **Custom Docker images come from `handle_docker`, not hard-coded tags.** Reference build/test images
   as `${{ fromJSON(needs.docker.outputs.images).ov_build.<name> }}` and add `Docker` to `needs`. Plain
   passthrough images must use the ACR mirror `openvinogithubactions.azurecr.io/...`, never `docker.io`.
7. **Keep changes scoped and validate before finishing** (see **Validation**). `.github/**` is owned by
   `@openvinotoolkit/openvino-ci-maintainers` and changes there are label `category: CI`
   (workflows also get `github_actions`).

## Key conventions

* **Runner selection** (`runs-on`) — self-hosted Azure pools `aks-{os}-{cores}-cores-{ram}gb[-arm]` for
  heavy build/test; GitHub-hosted (`ubuntu-22.04`, ...) for light jobs (labelers, style). GPU jobs use
  `[ self-hosted, gpu|igpu|dgpu ]` and **must** run in Docker. Azure `aks-*` runners are required to
  pull from the ACR / use custom images. Match cores to parallelism (see runners.md).
* **Containers** — most self-hosted jobs run in a `container:`. Mount the shared drive with
  `volumes: [ /mount:/mount ]` and add `${{ github.workspace }}:${{ github.workspace }}` where the
  workspace must be identical inside/outside the container.
* **Caches** — GHA cache (`actions/cache`, ≤10 GB) for small deps; shared drive (`/mount/...`, e.g.
  `PIP_CACHE_PATH: /mount/caches/pip/linux`) for large assets on Linux self-hosted; `sccache` → Azure
  Blob for C/C++ build cache (needs `SCCACHE_AZURE_*` env + `CMAKE_*_COMPILER_LAUNCHER: sccache` +
  `SCCACHE_AZURE_KEY_PREFIX`).
* **Artifacts** — `Build` job packs and uploads; test jobs `needs: Build` and download. Follow the
  existing `store_artifacts`/`restore_artifacts` actions and artifact-name conventions in the workflow.
* **`timeout-minutes`** — always set a sensible per-job timeout.

## Common tasks

### Add a test (choose the smallest unit)
Follow [adding_tests.md](../../../docs/dev/ci/github_actions/adding_tests.md):
* **New tests inside an existing suite** → no workflow change needed.
* **New step in an existing job** → add a `name` + `run` step, gate with
  `if: fromJSON(inputs.affected-components).<COMPONENT>.test` when component-specific.
* **New job** → add to the right validation workflow (or a `job_*.yml`); set `needs: [Build, Smart_CI]`,
  `runs-on`, `container`, `timeout-minutes`, and a Smart CI `if:`.

### Add / edit a reusable workflow (`job_*.yml`)
1. Give it `on: workflow_call:` with typed `inputs` (`runner`, `image`, `affected-components`,
   `python-version`, ...) and `permissions: read-all`.
2. Reference it from validation workflows via `uses: ./.github/workflows/job_<name>.yml` with `with:` and
   `needs: [ Build, Smart_CI ]`.
3. Keep the input contract minimal and documented via `description:`.

### Wire up Smart CI for a component
1. Map source paths → label in `.github/labeler.yml` (`'category: X': [globs]`).
2. Declare dependents in `.github/components.yml` under `revalidate:` (build+test) / `build:` (build
   only); use `[]` for none, or `'all'` to force full validation. Dependencies are **not** transitive.
3. Add `Smart_CI` to the validating job's `needs` and gate with
   `if: fromJSON(needs.smart_ci.outputs.affected_components).<COMPONENT>.{build,test}`. Keep the same
   condition on **every** step/job in the dependency chain — a skipped step feeding an ungated dependent
   leaves it running against missing outputs.
4. Aggregate results into the `Overall_Status` job (it `needs:` the real jobs and reports one required
   check). A workflow that must be **required** cannot use a `paths:` filter — a filtered-out run reports
   no status and blocks the merge queue; rely on Smart CI + `Overall_Status` instead.

### Add / change a custom action
* Composite `action.yml` under `.github/actions/<name>/`. Declare `inputs` (with `description`,
  `required`, `default`) and `runs: using: composite`. Every `run` step needs an explicit `shell:`.
* Reference untrusted input via `env:` inside the step, not inline interpolation.
* Update [custom_actions.md](../../../docs/dev/ci/github_actions/custom_actions.md) if the action is
  user-facing.

#### Choosing the implementation environment
* **Python is the default and preferred** implementation language for a custom action's logic. Use it for
  any action that does **not** need extensive access to the GitHub (Actions) API — file/artifact
  handling, environment setup, packaging, running tools, parsing, etc.
  * If the action needs a `requirements.txt`, it must pin the **full dependency tree**, not just
    top-level packages. Generate it from a clean environment with `pip freeze`:
    ```bash
    python3 -m venv /tmp/act-env && . /tmp/act-env/bin/activate
    pip install <top-level-deps>            # only the packages you directly import
    pip freeze > .github/actions/<name>/requirements.txt
    ```
    This makes installs reproducible and pinned. Regenerate the same way whenever deps change.
* **Use JavaScript/TypeScript only when the action uses the GitHub (Actions) API extensively** — the
  Octokit/`@actions/*` toolkit gives first-class typed access to it. The bundled
  [`.github/actions/cache`](../../../.github/actions/cache) action is the reference example of a
  JS-based action.
* When in doubt, prefer Python and keep API interaction minimal.

### Add / use a custom Docker image
Follow [docker_images.md](../../../docs/dev/ci/github_actions/docker_images.md): add a Dockerfile under
`.github/dockerfiles/{ov_build,ov_test}/<platform>/`, ensure a `Docker` job runs `handle_docker` with the
image path in `images:`, add `Docker` to consumers' `needs`, and set
`image: ${{ fromJSON(needs.docker.outputs.images).<group>.<name> }}`. When adding a new env-setup script,
add it under `category: docker_env` in `labeler.yml`, exclude it from `.dockerignore`, and bump
`.github/dockerfiles/docker_tag` (handle_docker prompts you).

### Change a runner or container
Pick the pool from [runners.md](../../../docs/dev/ci/github_actions/runners.md); keep `container`
volumes/options (shared drive, sccache) consistent with sibling jobs. GPU → Docker + `self-hosted` label.

## Validation before finishing

* **YAML/lint**: run `actionlint` if available (`actionlint .github/workflows/<file>.yml`); otherwise
  sanity-check YAML parses. Mirror what `workflows_scans.yml` (CodeQL `actions` + semgrep) and
  `dependency_review.yml` enforce — those run on any `.github/workflows/**` change.
* **Pinning**: every third-party `uses:` is a full SHA + version comment.
* **Permissions**: workflow defaults to least privilege; extra scopes are job-scoped and justified.
* **Smart CI**: new component-specific jobs/steps are gated and `Smart_CI` is in `needs`.
* **Scope**: only intended files changed; no stray `docker_tag`/labeler/components edits unless required.
* Do **not** hand-edit any `*.lock.yml` — those belong to agentic workflows.

## Pitfalls to check

* **Unpinned or tag-pinned third-party action** — supply-chain risk; CodeQL/semgrep will flag it.
* **Untrusted input in `run:`** — `${{ github.event.* }}` interpolated into shell is an injection vector;
  route through `env:`.
* **Over-broad `permissions:`** — especially `write` scopes at workflow level.
* **Job that ignores Smart CI** — burns limited self-hosted/GPU capacity on unaffected PRs.
* **Smart CI condition mismatch across a dependency chain** — a step skipped by a Smart CI `if:` whose
  dependent step/job lacks the same condition runs against missing outputs/artifacts. Gate the whole
  chain consistently.
* **`paths:` filter on a required workflow** — filtered-out runs report no status and hang the merge
  queue; use Smart CI + `Overall_Status` instead of `paths:`.
* **Hard-coded `docker.io` / non-ACR image on `aks-*`** — pulls fail or hit rate limits; use the ACR
  mirror or `handle_docker` output.
* **Missing `Docker`/`Smart_CI`/`Build` in `needs`** — races or missing artifacts/inputs.
* **Editing a `job_*.yml` input contract** without updating every caller's `with:` block.
* **Docker env change without `docker_tag` bump** — the image won't rebuild; handle_docker fails the check.
* **Missing `timeout-minutes`** — a hung job can occupy a runner indefinitely.
* **GPU job without Docker** — required on persistent GPU runners.

## Skill self-improvement

Keep this skill and the CI docs in sync with reality. When a change reveals a new rule, pattern, or
footgun:

* **New reusable workflow / custom action / Dockerfile convention** — note it under **Common tasks** and,
  if user-facing, in the matching page under `docs/dev/ci/github_actions/`.
* **New footgun** (a pinning/permission/Smart CI/Docker/cache mistake that bit you) — add it to
  **Pitfalls**.
* **A rule becomes obsolete** (a workflow removed, a runner pool renamed, a path moved) — update or
  remove the stale entry instead of leaving it.
* **Keep it concise** — prefer linking to the (updated) doc over duplicating detail; this file is a short
  actionable checklist, not a second copy of the documentation.
* This directory is reachable via both `.agents/skills/` and `.claude/skills/` (the former is a symlink to
  the latter), so a single edit updates both — do not create a duplicate copy.
