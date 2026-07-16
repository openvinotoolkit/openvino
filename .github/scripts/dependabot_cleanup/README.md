# Dependabot branch cleanup

Sometimes `dependabot/*` branches are not deleted automatically after their pull
request is merged or closed, leaving orphaned branches behind. This script finds
and removes those stale branches.

It is invoked on a daily schedule by
[`.github/workflows/cleanup_dependabot_branches.yml`](../../workflows/cleanup_dependabot_branches.yml)
and can also be triggered manually via `workflow_dispatch`.

## How the staleness check works

1. Enumerate every branch whose name starts with `dependabot/` (pagination
   handled automatically).
2. Collect the head branch names of all **open** pull requests.
3. A dependabot branch is considered **stale** (safe to delete) when it is *not*
   the head of any open PR. This covers branches whose PRs were merged or closed
   as well as branches that never had a PR.
4. Branches that still have an **open** PR are always kept and never deleted.

Stale branches are deleted through the git refs API. A failure to delete a
single branch is logged and does not abort the whole run; the script exits with
a non-zero status only if at least one real deletion attempt failed.

## Dry-run behavior

- `--dry-run` (default when run locally) only logs which branches *would* be
  deleted without deleting anything.
- `--no-dry-run` performs real deletions.
- When neither flag is passed, the default is taken from the `DRY_RUN`
  environment variable (`true`/`false`), falling back to `true` (safe) if unset.

The workflow wires this so that **scheduled cron runs perform real deletions**
while **manual `workflow_dispatch` runs default to a safe dry-run** (maintainers
can untick the `dry_run` input to actually delete).

## Running manually

```bash
pip install -r requirements.txt

# Preview only (safe):
GITHUB_TOKEN=<token> GITHUB_REPOSITORY=openvinotoolkit/openvino \
    python3 cleanup_dependabot_branches.py --dry-run

# Actually delete stale branches:
GITHUB_TOKEN=<token> GITHUB_REPOSITORY=openvinotoolkit/openvino \
    python3 cleanup_dependabot_branches.py --no-dry-run
```

The token needs `contents: write` permission on the repository to delete
branches.
