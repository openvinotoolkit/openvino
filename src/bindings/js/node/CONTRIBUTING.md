# Contributing to OpenVINO™ Node.js API

Your commitment to this project is greatly appreciated and the following guide is intended to help you contribute.  

Make sure to read [main contribution guide](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md) first. It covers most topics related to contributing to OpenVINO.


## TLDR

1. Decide what you want to change.
2. Create your fork of the OpenVINO repository.
3. Create a branch with a meaningful name for your changes.
4. Align the code style, commit the changes, and run tests.
5. Create a Pull Request, which clearly describes what has been changed and why.
6. Go through the Code Review.
7. Get your awesome code merged!

Read the section below for more details.


## How to Decide What to Change

In case of minor fixes, like changing variable names, additional parameter checks, etc., go to the next step.

However, if you want to bring significant changes, for example, the extension of architecture or a big part of functionality, that involves a large amount
of source code, open [an issue](https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=enhancement%2Cfeature&projects=&template=feature_request.yml&title=%5BFeature+Request%5D%3A+) first and discuss your idea with
codeowners. It will prevent you from doing extra work.

You can also take one of the well-described tasks from the [Good First Issue](https://github.com/orgs/openvinotoolkit/projects/3/views/14) section. It can be a great start to contributing with codeowners' support!


## Let's code

Get familiar with Node.js API architecture and code samples.
Refer to the [guide](../docs/code_examples.md), which will help you understand the component structure and the code style.

The environment setup and build instructions can be found in [Building the Node.js API](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/docs/README.md#openvino-node-package-developer-documentation).

Run tests! If you add a new functionality, make sure that it is covered by tests first.
Read [the guide](../docs/test_examples.md) for more details about the tests and their runs.
Many CI checks will run after getting a Code Review. Make sure that
all checks have passed. CI checks are composed of both functional tests and code-style checks and may fail because of warnings/errors in both stages.

Remember to follow [our codestyle](../docs/CODESTYLE.md).
By following the provided guide and using an automotive code style checking tool, like
**eslint** and **clang-format-9**, you will save some time and help with the code review of proposed changes.


## Description of the Pull Request

Append all PR titles with the `[OV JS]` tag. Provide any relevant details in the description, as it will definitely help with the review. The minimum requirement is a compact, bulleted list of proposed changes.

Use the following template:
```
*Describe what is the purpose of this PR*

### Details:
- *Describe your changes.*
- ...

```


## License

By contributing to the OpenVINO project, you agree that your contributions will be
licensed under the terms of the [LICENSE](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE).
