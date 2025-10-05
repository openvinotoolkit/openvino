# Wait for Check Completion Action

A GitHub Action that waits for multiple checks to complete before proceeding.

## Usage

```yaml
- name: Wait for checks to complete
  uses: ./.github/actions/wait-for-check-completion
  with:
    ref: ${{ github.event.pull_request.head.sha }}
    check-names: 'triage,build,test'
    repo-token: ${{ secrets.GITHUB_TOKEN }}
    wait-interval: 10
    timeout: 600
```

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `ref` | The git reference (commit SHA, branch, or tag) to check | Yes | - |
| `check-names` | Comma-separated list of check names to wait for (e.g., "check1,check2,check3") | Yes | - |
| `repo-token` | GitHub token for API access | Yes | - |
| `wait-interval` | Wait interval in seconds between status checks | No | `10` |
| `timeout` | Maximum timeout in seconds to wait for the check | No | `600` |
| `owner` | Repository owner (defaults to current repository owner) | No | Current repo owner |
| `repo` | Repository name (defaults to current repository name) | No | Current repo name |

## Outputs

| Output | Description |
|--------|-------------|
| `conclusion` | The overall conclusion of the checks (success, failure, action_required, mixed) |
| `status` | The overall status of the checks (completed, mixed) |
| `results` | JSON object containing results for each check with their individual status and conclusion |

## Examples

### Basic Usage

Wait for multiple checks to complete:

```yaml
- name: Wait for CI checks to complete
  uses: ./.github/actions/wait-for-check-completion
  with:
    ref: ${{ github.event.pull_request.head.sha }}
    check-names: 'build,test,lint'
    repo-token: ${{ secrets.GITHUB_TOKEN }}
```

### Single Check

Wait for a single check:

```yaml
- name: Wait for labeler to finish
  uses: ./.github/actions/wait-for-check-completion
  with:
    ref: ${{ github.event.pull_request.head.sha }}
    check-names: 'triage'
    repo-token: ${{ secrets.GITHUB_TOKEN }}
```

### Custom Settings

```yaml
- name: Wait for all CI to complete
  uses: ./.github/actions/wait-for-check-completion
  with:
    ref: ${{ github.sha }}
    check-names: 'continuous-integration,code-quality,security-scan'
    repo-token: ${{ secrets.GITHUB_TOKEN }}
    wait-interval: 30
    timeout: 1800
```

### Using Results Output

```yaml
- name: Wait for checks and analyze results
  id: wait-checks
  uses: ./.github/actions/wait-for-check-completion
  with:
    ref: ${{ github.event.pull_request.head.sha }}
    check-names: 'build,test,lint'
    repo-token: ${{ secrets.GITHUB_TOKEN }}

- name: Process results
  run: |
    echo "Overall conclusion: ${{ steps.wait-checks.outputs.conclusion }}"
    echo "Individual results: ${{ steps.wait-checks.outputs.results }}"
```

## Behavior

- The action waits for ALL specified checks to complete
- It polls all checks simultaneously and tracks their individual progress
- The overall conclusion is determined as follows:
  - `success`: All checks have successful conclusions (`success`)
  - `failure`: At least one check failed (`failure`, `cancelled`, `timed_out`)
  - `action_required`: At least one check requires action
  - `mixed`: Checks completed with mixed conclusions
- Individual check results are available in the `results` output as JSON
- If the timeout is reached before all checks complete, the action fails
- The action provides detailed logging of each check's progress

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

## Dependencies

* Uses `caniuse-lite` sourced from [caniuse.com](caniuse.com)
