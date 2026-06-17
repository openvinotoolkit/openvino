# How to Prepare a Good PR

   OpenVINO is an open-source project and you can contribute to its code directly. 
   To do so, follow these guidelines for creating Pull Requests, so that your 
   changes get the highest chance of being merged.


## General Rules of a Good Pull Request

* Create your own fork of the repository and use it to create PRs. 
  Avoid creating change branches in the main repository.
* Choose a proper branch for your work and create your own branch based on it. 
* Give your branches, commits, and Pull Requests meaningful names and descriptions. 
  It helps to track changes later. If your changes cover a particular component, 
  you can indicate it in the PR name as a prefix, for example: ``[DOCS] PR name``.
* Follow the [OpenVINO code style guide](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/coding_style.md).
* Make your PRs small - each PR should address one issue. Remove all changes 
  unrelated to the PR.
* [Link your Pull Request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#manually-linking-a-pull-request-to-an-issue-using-the-pull-request-sidebar) if it addresses one.
* Document your contribution! If your changes may impact how the user works with
  OpenVINO, provide the information in proper articles. You can do it yourself, 
  or contact one of OpenVINO documentation contributors to work together on
  developing the right content. 
* Complete the AI section in the [PR template](./.github/pull_request_template.md)
  for every PR and follow the [AI Usage Policy](./AI_USAGE_POLICY.md).
* For Work In Progress, or checking test results early, use a Draft PR.


## Ensure Change Quality

Your pull request will be automatically tested by OpenVINO™'s pre-commit and marked 
as "green" if it is ready for merging. If any builders fail, the status is "red," 
you need to fix the issues listed in console logs. Any change to the PR branch will 
automatically trigger the checks, so you don't need to recreate the PR, Just wait
for the updated results. 

Regardless of the automated tests, you should ensure the quality of your changes:

* Test your changes locally:
  * Make sure to double-check your code. 
  * Run tests locally to identify and fix potential issues (execute test binaries 
    from the artifacts directory, e.g. ``<source dir>/bin/intel64/Release/ieFuncTests``)
* Before creating a PR, make sure that your branch is up to date with the latest 
  state of the branch you want to contribute to (e.g. git fetch upstream && git 
  merge upstream/master).


## Branching Policy

* The "master" branch is used for development and constitutes the base for each new release.
* Each OpenVINO release has its own branch: ``releases/<year>/<release number>``.
* The final release each year is considered a Long Term Support version, 
  which means it remains active.
* Contributions are accepted only by active branches, which are:
  * the "master" branch for future releases,
  * the most recently published version for fixes,
  * LTS versions (for two years from their release dates).


## Pull Request Activity and Review Process

### What is considered activity

A Pull Request (PR) is considered **active** when there is meaningful progress, such as:

* Code updates addressing review feedback
* Technical discussion resolving open questions
* Significant rework or design clarification

Simple reminders, “ping” messages, or trivial updates (e.g. rebasing without changes) are not considered meaningful activity.


### Response expectations

After review feedback is provided:

* Contributors are expected to respond within **21 days**
* If there is no response, the PR may be considered **inactive**

This helps ensure that review bandwidth is focused on contributions that are actively progressing.


### Stale Pull Requests

If a PR becomes inactive:

* A maintainer may mark it with a `stale` label
* A reminder comment may be added requesting an update

This is a signal that the PR requires attention to continue review.


### Closing inactive Pull Requests

If there is no meaningful activity within **7 days after being marked as stale**:

* The PR may be **closed by maintainers**

Closing a PR due to inactivity:

* Is an **administrative action**, not a rejection
* Does **not prevent resubmission** of the contribution later


### Continuing or reviving work

If a PR has been closed due to inactivity:

* Contributors are welcome to reopen it (if possible), or
* Submit a new PR referencing the previous work

Maintainers will resume the review once activity continues.


### Maintainer continuation

To avoid losing valuable contributions:

* If a PR is inactive but relevant, maintainers may:
  * Continue work directly on top of the PR, or
  * Create a follow-up PR based on the original contribution

In such cases, credit to the original author will be preserved.


### Community reviews

OpenVINO encourages community participation in the review process.

* Any PRs are open for review by external contributors
* Constructive feedback from the community may be considered during merge decisions

Community contributions are highly valued and help improve review scalability.


## Need Additional Help? Check these Articles

* [How to create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) 
* [Install Git](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)
* If you want to add a new sample, please have a look at the Guide for contributing
  to C++/C/Python OV samples and add the license statement at the top of new files for
  C++ example, Python example.
