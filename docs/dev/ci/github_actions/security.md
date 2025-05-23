# Security best practices for GitHub Actions Workflows

There are a few simple steps that we should follow to ensure our workflows are not vulnerable to common attacks.

## Adjust `GITHUB_TOKEN` permissions

Use the `permissions` key to make sure the `GITHUB_TOKEN` is configured with the least privileges for each job.

Start with relatively safe permissions:

```yaml
permissions: read-all
```

If you need more permissions, declare them at the job level when possible, for example:

```yaml
jobs:
  stale:
    runs-on: ubuntu-latest

    # GITHUB_TOKEN will have only these permissions for
    # `stale` job
    permissions:
      issues: write
      pull-requests: write

    steps:
      - uses: actions/stale@f7176fd3007623b69d27091f9b9d4ab7995f0a06

```

Check [GitHub documentation](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/controlling-permissions-for-github_token) on this also.

## Reduce the scope of environment variables

Environment variables should be declared at the step level when possible (e.g. the variable is used only in this exact step). Only put variables on the job level when they're used by a few steps, and on the workflow level when they're used by most of the steps.

Example from [the official GitHub documentation](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/store-information-in-variables):

```yaml
name: Greeting on variable day

on:
  workflow_dispatch

# Workflow level variables. Avoid using these.
env:
  DAY_OF_WEEK: Monday

jobs:
  greeting_job:
    runs-on: ubuntu-latest
    # Job level variables
    env:
      Greeting: Hello
    steps:
      - name: "Say Hello Mona it's Monday"
        run: echo "$Greeting $First_Name. Today is $DAY_OF_WEEK!"
        # Step level variables. Prefer this approach
        env:
          First_Name: Mona

```

## Avoid using `pull_request_target`

**Never** use `pull_request_target` trigger event for workflows. If you want to use `pull_request_target`, contact a member of the OpenVINO GitHub Actions task force first. Check [GitHub blog post](https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/) on this as well.

## Handle secrets correctly

**Never ever** use plain-text secrets hard-coded in GitHub Actions Workflow. If you need to use secrets, contact a member of the OpenVINO GitHub Actions task force first.

## Be careful with user input.

Most of GitHub context variables propagated from user input. That means they should be treated as an untrusted and potentially malicious. There are some tactics you can use to mitigate the risk:
- Instead of using inline scripts, create an action and pass the variable as an argument
- Put the value into an environment variable for the step, and use the variable in the script

More details are available in [this](https://securitylab.github.com/resources/github-actions-untrusted-input/) blog post.

## Pin versions for GitHub Actions

When using third-party actions, pin the version with a commit hash rather than a tag to shield your workflow from potential supply-chain compromise.

For example, instead of this:

```yaml
uses: actions/checkout@v4.2.2
```

use this:

```yaml
uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
```

## Further reading
Follow general [recommendations from GitHub itself](https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions)
