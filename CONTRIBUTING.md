# How to contribute to the OpenVINO repository

We welcome community contributions to OpenVINO™. Please read the following guide to learn how to find ideas for contribution, follow best practices for pull requests, and test your changes with our established checks.


## Before you start contributing you should

-   Make sure you agree to contribute your code under  [OpenVINO™ (Apache 2.0) license](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE).
-   Decide what you’re going to contribute. If you are not sure what you want to work on, check out [Contributions Welcome](https://github.com/openvinotoolkit/openvino/issues/17502). See if there isn't anyone already working on the subject you choose, in which case you may still contribute, providing support and suggestions for the given issue or pull request.
-   If you are going to fix a bug, check if it still exists. You can do it by building the latest master branch and making sure that the error is still reproducible there. We do not fix bugs that only affect older non-LTS releases like 2020.2, for example (see more details about our [branching strategy](https://github.com/openvinotoolkit/openvino/wiki/Branches)).


## "Fork & Pull Request model" for code contribution

### [](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md#the-instruction-in-brief)The instruction in brief

-   Register at GitHub. Create your fork of the OpenVINO™ repository  [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)  (see  [https://help.github.com/articles/fork-a-repo](https://help.github.com/articles/fork-a-repo)  for details).
-   Install Git.
    -   Set your user name and email address in Git configuration according to the GitHub account (see  [First-Time-Git-Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)  for details).
-   Choose a task for yourself. It may be a bugfix or an entirely new piece of code.
-   Choose a base branch for your work. More details about branches and policies are here:  [Branches](https://github.com/openvinotoolkit/openvino/wiki/Branches)
-   Clone your fork to your computer.
-   Create a new branch (give it a meaningful name) from the base branch of your choice.
-   Modify / add the code, following our  [Coding Style Guide](./docs/dev/coding_style.md).
-   If you want to add a new sample, please have a look at the  [Guide for contributing to C++/C/Python IE samples](https://github.com/openvinotoolkit/openvino/wiki/SampleContribute)
-   If you want to contribute to the documentation and want to add a new guide, follow that instruction [Documentation guidelines](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLinesDocumentation)
-   Run testsuite locally:
    -   execute each test binary from the artifacts directory, e.g.  `<source dir>/bin/intel64/Release/ieFuncTests`
-   When you are done, make sure that your branch is up to date with latest state of the branch you want to contribute to (e.g.  `git fetch upstream && git merge upstream/master`). If so, push your branch to your GitHub fork and create a pull request from your branch to the base branch (see  [using-pull-requests](https://help.github.com/articles/using-pull-requests)  for details).

## Making a good pull request

Following these guidelines will increase the likelihood of your pull request being accepted:

-   One PR – one issue.
-   Build perfectly on your local system.
-   Choose the right base branch, based on our [Branch Guidelines](https://github.com/openvinotoolkit/openvino/wiki/Branches).
-   Follow the  [Coding Style Guide](./docs/dev/coding_style.md) for your code.
-   Document your contribution, if you decide it may benefit OpenVINO users. You may do it yourself by editing the files in the "docs" directory or contact someone working with documentation to provide them with the right information.
-   Cover your changes with test. 
-   Add the license statement at the top of new files [C++ example](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/classification_sample_async/main.cpp#L1-L2), [Python example](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/hello_classification/hello_classification.py#L3-L4). 
-   Add proper information to the PR: a meaningful title, the reason why you made the commit, and a link to the issue page, if it exists.
-   Remove changes unrelated to the PR.
-   If it is still WIP and you want to check CI test results early, use a  _Draft_  PR.
-   Submit your PR and become an OpenVINO™ contributor! 


## Testing and merging pull requests

Your pull request will be automatically tested by OpenVINO™'s precommit (testing statuses are automatically reported as "green" or "red" circles in precommit steps on the PR page). If any builders fail, you need to fix the issues before the PR can be merged. If you push any changes to your branch on GitHub the tests will re-run automatically. No need to close pull request and open a new one!


When an assigned reviewer accepts the pull request and the pre-commit is "green", the review status is set to "Approved", which informs OpenVINO™ maintainers that they can merge your pull request.