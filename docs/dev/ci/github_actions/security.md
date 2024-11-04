# Security best practices for GitHub Actions Workflows

There are a few simple steps that we should follow to increase ensure our workflows are not vulnerable to common attacks:

- Use the “permissions” key to make sure the GITHUB_TOKEN is configured with the least privileges for each job. Check [GitHub documentation](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/controlling-permissions-for-github_token) on this also.

- Declare environment variables at the step level when possible

- Never use `pull_request_target` trigger event for workflows. If you want to use `pull_request_target`, contact a member of OpenVINO GitHub Actions task force first. Check [GitHub blog post](https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/) on this as well.

- Never use plain-text secrets hard-coded in GitHub Actions Workflow. If you need to use secrets, contact a member of OpenVINO GitHub Actions task force first.

- Be careful with user input.

    Most of GitHub context variables propagated from user input, that means they should be treated as an untrusted and potentially malitious. There are some tactics you can use to mitigate the risk:
    - Instead of using inline scripts, create an action and pass the variable as an argument
    - Put the value into an environment variable for the step, and use the variable in the script

    More details are available in [this](https://securitylab.github.com/resources/github-actions-untrusted-input/) blog post.

- When using third-party actions, pin the version with a commit hash rather than a tag to shield your workflow from potential supply-chain compromise. 


- Follow general [recommendations from GitHub itself](https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions)
