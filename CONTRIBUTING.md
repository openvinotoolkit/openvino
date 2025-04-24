# Contributing to OpenVINO

## How to contribute to the OpenVINO project

OpenVINO™ is always looking for opportunities to improve and your contributions
play a big role in this process. There are several ways you can make the
product better.

# Table of Contents
1. [Forms of contribution](#Forms-of-contribution)
2. [Technical guide](#Technical-guide)


## Forms of contribution

### Provide Feedback

   * **Report bugs / issues**
     If you experience faulty behavior in OpenVINO or its components, you can
     [create a new issue](https://github.com/openvinotoolkit/openvino/issues)
     in the GitHub issue tracker.

   * **Propose new features / improvements**
     If you have a suggestion for improving OpenVINO or want to share your ideas, you can open a new
     [GitHub Discussion](https://github.com/openvinotoolkit/openvino/discussions).
     If your idea is already well defined, you can also create a
     [Feature Request Issue](https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=enhancement%2Cfeature&projects=&template=feature_request.yml&title=%5BFeature+Request%5D%3A+)
     In both cases, provide a detailed description, including use cases, benefits, and potential challenges.
     If your points are especially well aligned with the product vision, they will be included in the
     development roadmap.
     User feedback is crucial for OpenVINO development and even if your input is not immediately prioritized,
     it may be used at a later time or undertaken by the community, regardless of the official roadmap.


### Contribute Code Changes

   * **Fix Bugs or Develop New Features**
     If you want to help improving OpenVINO, choose one of the issues reported in
     [GitHub Issue Tracker](https://github.com/openvinotoolkit/openvino/issues) and
     [create a Pull Request](./CONTRIBUTING_PR.md) addressing it. If you want to start with something simple,
     check out the [first-time contributions section](#3-start-working-on-your-good-first-issue).
     If the feature you want to develop is more complex or not well defined by the reporter,
     it is always a good idea to [discuss it](https://github.com/openvinotoolkit/openvino/discussions)
     with OpenVINO developers first. Before creating a new PR, check if nobody is already
     working on it. In such a case, you may still help, having aligned with the other developer.

     Importantly, always check if the change hasn't been implemented before you start working on it!
     You can build OpenVINO using the latest master branch and make sure that it still needs your
     changes. Also, do not address issues that only affect older non-LTS releases, like 2022.2.

   * **Develop a New Device Plugin**
     Since the market of computing devices is constantly evolving, OpenVINO is always open to extending
     its support for new hardware. If you want to run inference on a device that is currently not supported,
     you can see how to develop a new plugin for it in the
     [Plugin Developer Guide](https://docs.openvino.ai/2025/documentation/openvino-extensibility/openvino-plugin-library.html).


### Improve documentation

   * **OpenVINO developer documentation** is contained entirely in this repository, under the
     [./docs/dev](https://github.com/openvinotoolkit/openvino/tree/master/docs/dev) folder.

   * **User documentation** is built from several sources and published at
     [docs.openvino.ai](https://docs.openvino.ai/), which is the recommended place for reading
     these documents. Use the files maintained in this repository only for editing purposes.

   * The easiest way to help with documentation is to review it and provide feedback on the
     existing articles. Whether you notice a mistake, see the possibility of improving the text,
     or think more information should be added, you can reach out to any of the documentation
     contributors to discuss the potential changes.

     You can also create a Pull Request directly, following the [editor's guide](./CONTRIBUTING_DOCS.md).


### Promote and Support OpenVINO

   * **Popularize OpenVINO**
     Articles, tutorials, blog posts, demos, videos, and any other involvement
     in the OpenVINO community is always a welcome contribution. If you discuss
     or present OpenVINO on various social platforms, you are raising awareness
     of the product among A.I. enthusiasts and enabling other people to discover
     the toolkit. Feel free to reach out to OpenVINO developers if you need help
     with making such community-based content.

   * **Help Other Community Members**
     If you are an experienced OpenVINO user and want to help, you can always
     share your expertise with the community. Check GitHub Discussions and
     Issues to see if you can help someone.

## Technical guide

This section lists all the necessary steps required to set up your environment, build OpenVINO locally, and run tests for specific components. It's a perfect place to start when you have just picked a Good First Issue and are wondering how to start working on it.

Keep in mind that we are here to help - **do not hesitate to ask the development team if something is not clear**. Such questions allow us to keep improving our documentation.

### 1. Prerequisites

You can start with the following links:
- [What is OpenVINO?](https://github.com/openvinotoolkit/openvino#what-is-openvino-toolkit)
- [OpenVINO architecture](https://github.com/openvinotoolkit/openvino/blob/master/src/docs/architecture.md)
- [User documentation](https://docs.openvino.ai/)
- [Blog post on contributing to OpenVINO](https://medium.com/openvino-toolkit/how-to-contribute-to-an-ai-open-source-project-c741f48e009e)
- [Pick up a Good First Issue](https://github.com/orgs/openvinotoolkit/projects/3)
- Check out [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG) - engage in discussions, ask questions and talk to OpenVINO developers

### 2. Building the project

In order to build the project, follow the [build instructions for your specific OS](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md).

### 3. Familiarize yourself with the component you'll be working with

Choose the component your Good First Issue is related to. You can run tests to make sure it works correctly.

##### APIs
- [C API](https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/c)
- [Core](https://github.com/openvinotoolkit/openvino/tree/master/src/core)
- [Python API](https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/python)
- [Node.js API](https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/js/node)

##### Frontends
- [IR Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/ir)
- [ONNX Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/onnx)
- [PaddlePaddle Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle)
- [PyTorch Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/pytorch)
- [TensorFlow Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/tensorflow)

##### Plugins
- [Auto plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/auto)
- [CPU plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu)
- [GPU plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu)
- [NPU plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu)
- [Hetero plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/hetero)
- [Template plugin](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/template)

##### Tools
- [Benchmark Tool](https://github.com/openvinotoolkit/openvino/tree/master/tools/benchmark_tool)
- [OpenVINO Model Converter](https://github.com/openvinotoolkit/openvino/tree/master/tools/ovc)

##### Others
- [Documentation](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_DOCS.md)

### 3. Start working on your Good First Issue

To start contributing, pick a task from the [Good First Issues board](https://github.com/orgs/openvinotoolkit/projects/3).

To be assigned to an issue, simply leave a comment with the `.take` command in the selected issue.
Use the issue description and build OpenVINO locally to complete the task.

You can always ask users tagged in the "Contact points" section for help!
Visit [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG) and ask
questions in the channel dedicated to Good First Issue support.

### 4. Submit a PR with your changes

Follow our [Good Pull Request guidelines](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_PR.md). Please remember about [linking your Pull Request to the issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#manually-linking-a-pull-request-to-an-issue-using-the-pull-request-sidebar) it addresses.

### 5. Wait for a review

We'll make sure to review your Pull Request as soon as possible and provide you with our feedback. You can expect a merge once your changes are validated with automatic tests and approved by maintainers.

## License

By contributing to the OpenVINO project, you agree that your contributions will be
licensed under the terms stated in the [LICENSE](./LICENSE) file.
