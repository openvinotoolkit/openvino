# Contributing to OpenVINO™ Node.js API

If you look at this guide, we already appreciate you. We try to support you
on the way to contributing to this project.

First of all, read [main contribution guide](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md). It covers most topics related to contributing to OpenVINO.


## TLDR

- Decide what you want to change
- Create your fork of the openvino repository
- Implement changes, align code style, run tests
- Keep changes in the branch with a clear name
- Create a Pull Request, clearly describe what has been changed and why
- Go through Code Review
- Be merged, you are awesome!

Detailed explanation below:


## How to decide what to change

It can be minor fixes, like changing variable names, additional parameter checks, etc. In this case, go ahead to the next step.

If you want to bring significant changes, for example, the extension of architecture or a big part of functionality, that touches a lot
of source code, please, open [an issue](https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=enhancement%2Cfeature&projects=&template=feature_request.yml&title=%5BFeature+Request%5D%3A+) first and discuss your idea with
codeowners. It prevents you from doing extra work.

Or take one of the well-described tasks from the [Good First Issue](https://github.com/orgs/openvinotoolkit/projects/3/views/14) section. It can be a great start to contribution with codeowners support!


## Let's code

Be familiar with Node.js API architecture and code samples.
This [well-written guide](../docs/code_examples.md) helps to understand the component structure and code style that we follow.

The environment setup and building instructions can be found in [Building the Node.js API](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/docs/README.md#openvino-node-package-developer-documentation).

Run tests! If you added some new functionality, please, make sure that it is covered by tests.
[This guide](../docs/test_examples.md) helps be familiar with our tests and their run.
Many CI checks will run after getting a Code Review. Make sure that
all checks are green. CI checks are composed of both functional tests and code-style checks and may fail because of warnings/errors in both stages.

Make sure that you follow [our codestyle](../docs/CODESTYLE.md).
By following the provided guide and using an automotive code style checking tool, like
**eslint** and **clang-format-9** save you and the reviewers time to perform a code review of
proposed changes.


## Description of the Pull Request

Please append all PR titles with the tag `[OV JS]`. Feel free to describe any level of relevant details in the PR, it helps a lot with the review process. The minimum requirement is a compact description of changes made, the form of a bullet-point list is appreciated.

Template for contributors:
```
*What was the purpose of this PR*

### Details:
- *what you've done*
- ...

```


## License

By contributing to the OpenVINO project, you agree that your contributions will be
licensed under the terms stated in the [LICENSE](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE) file.
