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
* Document your contribution! If your changes may impact how the user works with
  OpenVINO, provide the information in proper articles. You can do it yourself, 
  or contact one of OpenVINO documentation contributors to work together on
  developing the right content. 
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


## Need Additional Help? Check these Articles

* [How to create a fork](https://help.github.com/articles/fork-a-rep) 
* [Install Git](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)
* If you want to add a new sample, please have a look at the Guide for contributing
  to C++/C/Python IE samples and add the license statement at the top of new files for
  C++ example, Python example.
