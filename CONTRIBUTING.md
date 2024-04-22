# Contributing to OpenVINO

OpenVINO™ always thrives to improve, and user contributions are one of the opportunities
that are crucial for this process. Read on to find out how you can participate.

> **NOTE**: By contributing to the OpenVINO project, you agree that your contributions will be
licensed under [the terms](./LICENSE).

## Forms of contribution

### Provide feedback

   * **Report bugs / issues**
     If you notice unexpected behavior in OpenVINO or its components, you can
     [create a new issue](https://github.com/openvinotoolkit/openvino/issues)
     in the GitHub issue tracker.

   * **Propose new features / improvements**
     If you want to share your ideas for improving OpenVINO, you can open a new
     [GitHub Discussion](https://github.com/openvinotoolkit/openvino/discussions).
     If your idea is already well defined, you can also create a
     [Feature Request Issue](https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=enhancement%2Cfeature&projects=&template=feature_request.yml&title=%5BFeature+Request%5D%3A+)
     In any case, you need to provide a detailed description, including use cases, benefits,
     and potential challenges.

### Contribute code changes

First of all, always start by checking if the change has not been implemented before!
Build OpenVINO, using the latest master branch and verify that your changes are still
applicable. If somebody else is already working on it, you may still help, having
aligned with the other developer. Also, do not address issues that only affect older
non-LTS releases, like 2022.2.

   * **Fix bugs**

     If you want to help improving OpenVINO, choose one of the issues reported in
     [GitHub Issue Tracker](https://github.com/openvinotoolkit/openvino/issues) and
     [create a Pull Request](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_PR.md)
     addressing it. Consider one of the tasks listed as
     [first-time contributions](https://github.com/orgs/openvinotoolkit/projects/3)

   * **Develop new features**

     If the feature you want to develop is more complex or not well defined yet,
     you may want to [discuss it](https://github.com/openvinotoolkit/openvino/discussions)
     with OpenVINO developers first.

   * **Develop a new device plugin**

     Since the market of computing devices is constantly evolving, OpenVINO is always open
     to extending its support for new hardware. If you want to run inference on a device that
     is currently not supported, you can see how to develop a new plugin for it in the
     [Plugin Developer Guide](https://docs.openvino.ai/nightly/documentation/openvino-extensibility/openvino-plugin-library.html).

### Improve documentation

The easiest way to help with documentation is to review it and provide feedback on the
existing articles. Whether you notice a mistake, see the possibility of improving the text,
or think more information should be added, you can reach out to any of the
[documentation maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-docs-maintainers)
to discuss the potential changes. You can also edit the articles directly on GitHub or open
them with [Github.dev](https://github.dev/openvinotoolkit/openvino). Do not forget to create
a **Pull Request**, following the [guidelines](./CONTRIBUTING_DOCS.md).

   * **Review user documentation**

     The documentation is built from several sources, mainly
     [docs](./docs) folder, using [Sphinx](https://www.sphinx-doc.org/) and the
     [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)
     markup language. It is published at [docs.openvino.ai](https://docs.openvino.ai/),
     which is recommended for reading the articles.

   * **Check OpenVINO developer documentation**

     The documentation is contained entirely in the [docs/dev](./docs/dev) folder.

### Promote and support OpenVINO

   * **Popularize OpenVINO**

     Articles, tutorials, blog posts, demos, videos, and any other involvement
     in the OpenVINO community is always a welcome contribution. If you discuss
     or present OpenVINO on various social platforms, you are raising awareness
     of the product among A.I. enthusiasts and enabling other people to discover
     the toolkit. Feel free to reach out to OpenVINO developers if you need help
     with making such community-based content.

   * **Help other community members**

     If you are an experienced OpenVINO user and want to help, you can share your expertise
     with the community at any time. Check GitHub
     [Discussions](https://github.com/openvinotoolkit/openvino/discussions) and
     [Issues](https://github.com/openvinotoolkit/openvino/issues)
     to see if you can help someone.

## Technical guide

This section will start you off with a few simple steps to begin your contribution.
In case of any doubts, you can talk to
[the development team](https://github.com/orgs/openvinotoolkit/teams/openvino-developers/teams).
Remember, your questions help us keep improving OpenVINO.

### 1. Build OpenVINO.

In order to build OpenVINO, follow the [build instructions for your specific OS](./docs/dev/build.md).

### 2. Choose a [Good First Issue](https://github.com/orgs/openvinotoolkit/projects/3)

Use the links below to explore the component in question. You can run tests to make sure it works correctly.

   * **APIs**
     - [C API](https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/c)
     - [Core](https://github.com/openvinotoolkit/openvino/tree/master/src/core)
     - [Python API](https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/python)

   * **Frontends**
     - [IR Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/ir)
     - [ONNX Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/onnx)
     - [PaddlePaddle Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle)
     - [PyTorch Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/pytorch)
     - [TensorFlow Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/tensorflow)

   * **Plugins**
     - [Auto plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/auto)
     - [CPU plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu)
     - [GPU plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu)
     - [NPU plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu)
     - [Hetero plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/hetero)
     - [Template plugin](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/template)

   * **Tools**
     - [Benchmark Tool](https://github.com/openvinotoolkit/openvino/tree/master/tools/benchmark_tool)
     - [Model Optimizer](https://github.com/openvinotoolkit/openvino/tree/master/tools/mo)

   * **Others**
     - [Documentation](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_DOCS.md)

### 3. Begin working on a "Good First Issue" or create a [new one](https://github.com/openvinotoolkit/openvino/issues/new?assignees=&labels=good+first+issue%2Cno_stale&projects=&template=good_first_issue.yml&title=%5BGood+First+Issue%5D%3A+)

Use the locally built OpenVINO and the information found in the issue description. Remember
that you can assign users in the **"Contact points"** section for help. You can also
visit [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG) and ask questions
in the channel dedicated to **"Good First Issue"** support.

### 4. Submit a PR with your changes

Follow the [Good Pull Request guidelines](./CONTRIBUTING_PR.md)
and do not forget to [link your Pull Request to the issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#manually-linking-a-pull-request-to-an-issue-using-the-pull-request-sidebar)
it addresses.

### 5. Wait for a review

We will make sure to review your **Pull Request** as soon as possible and provide feedback.
You can expect a merge once your changes have been validated with automatic tests and
approved by maintainers.

### Additional Resources

- [User documentation](https://docs.openvino.ai/)
- [Choose a Good First Issue](https://github.com/orgs/openvinotoolkit/projects/3)
- Learn more about [OpenVINO architecture](./src/docs/architecture.md)
- Check out a [blog post on contributing to OpenVINO](https://medium.com/openvino-toolkit/how-to-contribute-to-an-ai-open-source-project-c741f48e009e)
- Visit [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG) to join discussions and talk to OpenVINO developers.
