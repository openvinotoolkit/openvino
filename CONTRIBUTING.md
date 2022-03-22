# How to contribute to the OpenVINO repository

We welcome community contributions to OpenVINO™. Please read the following guide to learn how to find ideas for contribution, practices for good pull requests, checking your changes with our tests and more.


## Before you start contributing you should

-   Make sure you agree to contribute your code under  [OpenVINO™ (Apache 2.0)](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE)  license.
-   Figure out what you’re going to contribute. If you don’t know what you are going to work on, navigate to the   [Github "Issues" tab](https://github.com/openvinotoolkit/openvino/issues). Make sure that there isn't someone working on it. In the latter case you might provide support or suggestion in the issue or in the linked pull request.
-   If you are going to fix a bug, check that it's still exists in the latest release. This can be done by building the latest master branch, and make sure that the error is still reproducible there. We do not fix bugs that only affect older non-LTS releases like 2020.2 for example (more details about  [branching strategy](https://github.com/openvinotoolkit/openvino/wiki/Branches)).


## "Fork & Pull Request model" for code contribution

### [](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md#the-instruction-in-brief)The instruction in brief

-   Register at GitHub. Create your fork of OpenVINO™ repository  [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)  (see  [https://help.github.com/articles/fork-a-repo](https://help.github.com/articles/fork-a-repo)  for details).
-   Install Git.
    -   Set your user name and email address in a Git configuration according to GitHub account (see  [https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)  for details).
-   Choose a task for yourself. It could be a bugfix or some new code.
-   Choose a base branch for your work. More details about branches and policies are here:  [Branches](https://github.com/openvinotoolkit/openvino/wiki/Branches)
-   Clone your fork to your computer.
-   Create a new branch (with a meaningful name) from the base branch you chose.
-   Modify / add the code following our  [Coding Style Guide](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLines).
-   If you want to add a new sample, please look at this  [Guide for contributing to C++/C/Python IE samples](https://github.com/openvinotoolkit/openvino/wiki/SampleContribute)
-   If you want to contribute to the documentation and want to add a new guide, follow that instruction [Documentation guidelines](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLinesDocumentation)
-   Run testsuite locally:
    -   execute each test binary from the artifacts directory, e.g.  `<source dir>/bin/intel64/Release/ieFuncTests`
-   When you are done, make sure that your branch is to date with latest state of the branch you want to contribute to (e.g.  `git fetch upstream && git merge upstream/master`), push your branch to your GitHub fork; then create a pull request from your branch to the base branch (see  [https://help.github.com/articles/using-pull-requests](https://help.github.com/articles/using-pull-requests)  for details).

## Making a good pull request

Following these guidelines will increase the likelihood of your pull request being accepted:

-   One PR – one issue.
-   Build perfectly on your local system.
-   Choose the right base branch [Branches](https://github.com/openvinotoolkit/openvino/wiki/Branches).
-   Follow the  [Coding Style Guide](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLines) for your code.
-   Update documentation using [Documentation guidelines](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLinesDocumentation) if needed.
-   Cover your changes with test. 
-   Add license at the top of new files [C++ example](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/classification_sample_async/main.cpp#L1-L2), [Python example](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/hello_classification/hello_classification.py#L3-L4). 
-   Add enough information: a meaningful title, the reason why you made the commit and a link to the issue page if exists.
-   Remove unrelated to PR changes.
-   If it is still WIP and you want to check CI test results early then use  _Draft_  PR.
-   Submit your PR and become an OpenVINO™ contributor! 


## Testing and merging pull requests

Your pull request will be automatically tested by OpenVINO™'s precommit (testing status are automatically reported as "green" or "red" circles in precommit steps on PR's page). If any builders have failed, you need fix the issue. To rerun the automatic builds just push changes to your branch on GitHub. No need to close pull request and open a new one!


## Merging PR

As soon as the reviewer is fine with the pull request and precommit shows "green" status, the "Approved" review status is put, which signals OpenVINO™ maintainers that they can merge your pull request.