# Wait for Check Completion Action

A GitHub Action that waits for a specific check to complete before proceeding. This is useful for workflows that need to wait for other checks or workflows to finish.

## Usage

```yaml
- name: Wait for checks to complete
  uses: ./.github/actions/wait-for-check-completion
  with:
    ref: ${{ github.event.pull_request.head.sha }}
    check-name: 'triage'
    repo-token: ${{ secrets.GITHUB_TOKEN }}
    wait-interval: 10
    timeout: 600
```

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `ref` | The git reference (commit SHA, branch, or tag) to check | Yes | - |
| `check-name` | The name of the check to wait for | Yes | - |
| `repo-token` | GitHub token for API access | Yes | - |
| `wait-interval` | Wait interval in seconds between status checks | No | `10` |
| `timeout` | Maximum timeout in seconds to wait for the check | No | `600` |
| `owner` | Repository owner (defaults to current repository owner) | No | Current repo owner |
| `repo` | Repository name (defaults to current repository name) | No | Current repo name |

## Outputs

| Output | Description |
|--------|-------------|
| `conclusion` | The conclusion of the check (success, failure, neutral, cancelled, timed_out, action_required, stale, skipped) |
| `status` | The status of the check (queued, in_progress, completed) |

## Examples

### Basic Usage

Wait for a specific check to complete:

```yaml
- name: Wait for labeler to finish
  uses: ./.github/actions/wait-for-check-completion
  with:
    ref: ${{ github.event.pull_request.head.sha }}
    check-name: 'triage'
    repo-token: ${{ secrets.GITHUB_TOKEN }}
```

### Custom Wait Interval and Timeout

```yaml
- name: Wait for CI to complete
  uses: ./.github/actions/wait-for-check-completion
  with:
    ref: ${{ github.sha }}
    check-name: 'continuous-integration'
    repo-token: ${{ secrets.GITHUB_TOKEN }}
    wait-interval: 30
    timeout: 1800
```

### Conditional Wait

```yaml
- name: Wait for checks (PR only)
  uses: ./.github/actions/wait-for-check-completion
  if: ${{ github.event_name == 'pull_request' }}
  with:
    ref: ${{ github.event.pull_request.head.sha }}
    check-name: 'build'
    repo-token: ${{ secrets.GITHUB_TOKEN }}
```

## Behavior

- The action polls the GitHub API at regular intervals (specified by `wait-interval`) to check the status of the specified check
- It waits until the check status becomes `completed`
- The action succeeds if the check conclusion is `success`, `neutral`, or `skipped`
- The action fails if the check conclusion is `failure`, `cancelled`, or `timed_out`
- The action shows a warning for `action_required` or unknown conclusions
- If the timeout is reached before the check completes, the action fails

## Development

### Building

To build the action after making changes:

```bash
cd .github/actions/wait-for-check-completion
npm install
npm run build
```

### Testing

```bash
npm test
```
