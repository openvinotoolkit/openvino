# How to contribute to the OpenVINO repository

We suppose that you are an enthusiastic coder, want to contribute some code. For that purpose OpenVINO project now has a repository on the GitHub, to simplify everybody's life! All the bug fixes, new functionality, new tutorials etc. should be submitted via the GitHub's mechanism of pull requests.

If you are not familiar with the mechanism - do not worry, it's very simple. Keep reading.

## Before you start contributing you should

-   Make sure you agree to contribute your code under  [OpenVINO (Apache 2.0)](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE)  license.
-   If you are submitting a new module, you should go into  [openvino_contrib](https://github.com/openvinotoolkit/openvino_contrib)  repository by default.
-   If you are going to fix a bug, check that it's still exists. This can be done by building the latest  [releases/2020/3](https://github.com/openvinotoolkit/openvino/tree/releases/2020/3)  branch (LTS release) or the latest master branch, and make sure that the error is still reproducible there. We do not fix bugs that only affect older non-LTS releases like 2020.2 for example (more details about  [branching strategy](https://github.com/openvinotoolkit/openvino/wiki/Branches))
-   Make sure that nobody beat you into fixing or reporting the issue by doing a search on the  [Github OpenVINO issues](https://github.com/openvinotoolkit/openvino/issues)  page, and making sure that there isn't someone working on it. In the latter case you might provide support or suggestion in the issue or in the linked pull request.
-   If you have a question about the software, then this is  **NOT**  the right place. You should open up a question at the  [OpenVINO forum](https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/bd-p/distribution-openvino-toolkit). In order to post a decent question from the start, feel free to read the official forum guidelines.

Before you open up anything on the OpenVINO GitHub page, be sure that you are at the right place with your problem.

## "Fork & Pull Request model" for code contribution

### [](https://github.com/openvinotoolkit/openvino/wiki/Contribute#the-instruction-in-brief)The instruction in brief

-   Register at GitHub. Create your fork of OpenVINO repository  [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)  (see  [https://help.github.com/articles/fork-a-repo](https://help.github.com/articles/fork-a-repo)  for details).
-   Install Git.
    -   Set your user name and email address in a Git configuration according to GitHub account (see  [https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)  for details).
-   Choose a task for yourself. It could be a bugfix or some new code.
-   Choose a base branch for your work. More details about branches and policies are here:  [Branches](https://github.com/openvinotoolkit/openvino/wiki/Branches)
-   Clone your fork to your computer.
-   Create a new branch (with a meaningful name) from the base branch you chose.
-   Modify / add the code following our  [Coding Style Guide](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLines)  and  [Documentation guidelines](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLinesDocumentation).
-   If you want to add a new sample, please look at this  [Guide for contributing to C++/C/Python IE samples](https://github.com/openvinotoolkit/openvino/wiki/SampleContribute)
-   Run testsuite locally:
    -   execute each test binary from the artifacts directory, e.g.  `<source dir>/bin/intel64/Release/ieFuncTests`
-   If you contribute to the documentation and want to add a new guide:
    -   Create a new markdown file in an appropriate folder.
    -   **REQUIRED:**  The document title must contain a document label in a form:  `{#openvino_docs_<name>}`. For example:  `Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™ {#openvino_docs_MO_DG_IR_and_opsets}`.
    -   Add your file to the documentation structure. Open the documentation structure file  [`docs/doxygen/ie_docs.xml`](https://github.com/openvinotoolkit/openvino/blob/master/docs/doxygen/ie_docs.xml)  and add your file path to the appropriate section.
-   When you are done, make sure that your branch is to date with latest state of the branch you want to contribute to (e.g.  `git fetch upstream && git merge upstream/master`), push your branch to your GitHub fork; then create a pull request from your branch to the base branch (see  [https://help.github.com/articles/using-pull-requests](https://help.github.com/articles/using-pull-requests)  for details).

## Making a good pull request

Following these guidelines will increase the likelihood of your pull request being accepted:

-   Before pushing your PR to the repository, make sure that it builds perfectly fine on your local system.
-   Add enough information, like a meaningful title, the reason why you made the commit and a link to the issue page if you opened one for this PR.
-   Scope your PR to one issue. Before submitting, make sure the diff contains no unrelated changes. If you want to cover more than one issue, submit your changes for each as separate pull requests.
-   If you have added new functionality, you should update/create the relevant documentation, as well as add tests for it to the testsuite.
-   Try not to include "oops" commits - ones that just fix an error in the previous commit. If you have those, then before submitting  [squash](https://github.com/openvinotoolkit/openvino/wiki/Contribute#https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History#Squashing-Commits)  those fixes directly into the commits where they belong.
-   Make sure to choose the right base branch and to follow the  [Coding Style Guide](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLines)  for your code or  [Documentation guidelines](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLinesDocumentation)  you are changing documentation files.
-   Make sure to add test for new functionality or test that reproduces fixed bug with related test data. Please do not add extra images or videos, if some of existing media files are suitable.

## Testing and merging pull requests

-   Your pull request will be automatically tested by OpenVINO's precommit (testing status are automatically reported as "green" or "red" circles in precommit steps on PR's page). If any builders have failed, you should fix the issue. To rerun the automatic builds just push changes to your branch on GitHub. No need to close pull request and open a new one!
-   Once all the builders are "green", one of OpenVINO developers will review your code. Reviewer could ask you to modify your pull request. Please provide timely response for reviewers (within weeks, not months), otherwise you submission could be postponed or even rejected.

## PR review good practices

-   Originator is responsible for driving the review of changes and should ping reviewers periodically.
-   Originator should close comments from the Reviewer when it is resolved. The Reviewer may re-open the comment if he does not agree with the resolution.
-   Originator should request re-review from the Reviewer when all comments are resolved by pushing the button in the “Reviewers” section.
-   If it is still WIP and you want to check CI test results early then use  _Draft_  PR.
-   Do  **NOT**  rewrite history (push -f) once you converted draft PR into regular one, add new commits instead. Looking at diffs makes review easier.
-   Write meaningful description of commits resulting from review.  _"Addressing review comments"_  is  **NOT**  a good description! Having a quick look at good descriptions can tell you much what is going on in PR without a need to go through all of resolved comments.

## Merging PR

As soon as the reviewer is fine with the pull request and Precommit likes your code and shows "green" status, the "Approved" review status is put, which signals OpenVINO maintainers that they can merge your pull request.

© Copyright 2018-2022, OpenVINO team