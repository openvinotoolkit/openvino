# How to contribute to the OpenVINO repository

We welcome community contributions to OpenVINO™. Read the following guide to learn how to find ideas for contribution, practices for good pull requests, checking your changes with our tests, and more.


## Before you start contributing you should

-   Make sure you agree to contribute your code under [OpenVINO™ (Apache 2.0)](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE) license.
-   Figure out what contribution you are going to make. If you don’t know what you are going to work on, navigate to the [Github "Issues" tab](https://github.com/openvinotoolkit/openvino/issues) and find an open issue. Make sure no one is working on the issue. In the latter case, you can provide support or suggestion directly in the issue or in a linked pull request.
-   If you want to fix a bug, check that it still exists in the latest release. To do this, build the latest master branch and make sure that the error is still reproducible on the latest master. We do not fix bugs that affect only older non-LTS releases, for example, 2020.2. Read more about [branching strategy](https://github.com/openvinotoolkit/openvino/wiki/Branches)).


## "Fork & Pull Request model" for code contribution

### [](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md#the-instruction-in-brief)The instruction in brief

-   Register on GitHub. Create your fork of the OpenVINO™ repository [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino). See [https://help.github.com/articles/fork-a-repo](https://help.github.com/articles/fork-a-repo) for details about GitHub forks.
-   Install Git.
    -   Set your user name and email address in a Git configuration according to your GitHub account. See [https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup) for details about account settings.
-   Choose a task. It could be a bug fix or some new code.
-   Choose a base branch for your work. See [Branches](https://github.com/openvinotoolkit/openvino/wiki/Branches) for details about branches and policies.
-   Clone your fork to your computer.
-   Create a new branch with a meaningful name from the selected base branch.
-   Modify / add the code following our [Coding Style Guide](./docs/dev/coding_style.md).
-   If you want to add a new sample, read the [Guide for contributing to C++/C/Python IE samples](https://github.com/openvinotoolkit/openvino/wiki/SampleContribute).
-   If you want to contribute to the documentation and add a new guide, read the [Documentation guidelines](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLinesDocumentation).
-   Run testsuite locally:
    -   execute each test binary from the artifacts directory, for example `<source dir>/bin/intel64/Release/ieFuncTests`.
-   When you are done, make sure that your branch is up to date with the latest state of the branch you want to contribute to. For example, use `git fetch upstream && git merge upstream/master` commands. Push your branch to your GitHub fork. Then create a pull request from your branch to the base branch. See [https://help.github.com/articles/using-pull-requests](https://help.github.com/articles/using-pull-requests) for details about pull requests.

## Making a good pull request

Follow the guidelines to make sure that your pull request is accepted as soon as possible:

-   One PR – one issue.
-   Check the build on your local system.
-   Choose the right base branch [Branches](https://github.com/openvinotoolkit/openvino/wiki/Branches).
-   Follow the [Coding Style Guide](./docs/dev/coding_style.md) for your code.
-   Update documentation using [Documentation guidelines](https://github.com/openvinotoolkit/openvino/wiki/CodingStyleGuideLinesDocumentation), if needed.
-   Cover your changes with tests. 
-   Add license at the top of new files [C++ example](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/classification_sample_async/main.cpp#L1-L2), [Python example](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/hello_classification/hello_classification.py#L3-L4). 
-   Add the necessary information: a meaningful title, the reason why you made the commit, and a link to the issue page, if it exists.
-   Remove changes not related to the PR.
-   If the PR is still WIP and you want to check CI test results early, use _Draft_ PR.
-   Submit your PR and become an OpenVINO™ contributor! 


## Testing and merging pull requests

Your pull request will be automatically tested by the OpenVINO™ precommit. Testing status is automatically reported as "green" or "red" circles in the pre-commit steps on the PR's page. You need to fix failing builds to merge your PR. To rerun the automatic builds, push changes to your branch on GitHub. No need to close the pull request and open a new one!


## Merging PR

When the reviewer accepts the pull request and the precommit shows a "green" status, the review status is set to "Approved", which signals to the OpenVINO™ maintainers that they can merge your pull request.
