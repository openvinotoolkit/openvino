Technical Guide for Contribution
================================

This section will start you off with a few simple steps to begin your code contribution.
In case of any doubts, you can talk to
`the development team <https://github.com/orgs/openvinotoolkit/teams/openvino-developers/teams>`__.
Remember, your questions help us keep improving OpenVINO.

1. **Build OpenVINO.**

   In order to build OpenVINO, follow the
   `build instructions for your specific OS <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__.

2. **Choose a** `"Good First Issue" <https://github.com/orgs/openvinotoolkit/projects/3>`__.

   Use the links below to explore the component in question. You can run tests to make sure it works correctly.

   .. tab-set::

      .. tab-item:: APIs

         - `C API <https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/c>`__
         - `Core <https://github.com/openvinotoolkit/openvino/tree/master/src/core>`__
         - `Python API <https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/python>`__

      .. tab-item:: Frontends

         - `IR Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/ir>`__
         - `ONNX Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/onnx>`__
         - `PaddlePaddle Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle>`__
         - `PyTorch Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/pytorch>`__
         - `TensorFlow Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/tensorflow>`__

      .. tab-item:: Plugins

         - `Auto plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/auto>`__
         - `CPU plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu>`__
         - `GPU plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu>`__
         - `Hetero plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/hetero>`__
         - `Template plugin <https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/template>`__

      .. tab-item:: Tools

         - `Benchmark Tool <https://github.com/openvinotoolkit/openvino/tree/master/tools/benchmark_tool>`__
         - `Model Optimizer <https://github.com/openvinotoolkit/openvino/tree/master/tools/mo>`__

      .. tab-item:: Documentation

         - `Documentation <https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_DOCS.md>`__

3. **Begin working on a "Good First Issue" or create**
   `a new one <https://github.com/openvinotoolkit/openvino/issues/new?assignees=&labels=good+first+issue%2Cno_stale&projects=&template=good_first_issue.yml&title=%5BGood+First+Issue%5D%3A+>`__.

   Use the locally built OpenVINO and the information found in the issue description. Remember
   that you can assign users in the **"Contact points"** section for help. You can also
   visit `Intel DevHub Discord server <https://discord.gg/7pVRxUwdWG>`__ and ask questions
   in the channel dedicated to **"Good First Issue"** support.

4. **Submit a PR with your changes.**

   Follow the `guidelines <https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_PR.md>`__
   and do not forget to `link your Pull Request to the issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#manually-linking-a-pull-request-to-an-issue-using-the-pull-request-sidebar>`__
   it addresses.

5. **Wait for a review.**

   We will make sure to review your **Pull Request** as soon as possible and provide feedback.
   You can expect a merge once your changes have been validated with automatic tests and
   approved by maintainers.


Additional Resources
#####################

- Choose a `"Good First Issue" <https://github.com/orgs/openvinotoolkit/projects/3>`__.
- Learn more about `OpenVINO architecture <https://github.com/openvinotoolkit/openvino/blob/master/src/docs/architecture.md>`__.
- Check out a `blog post on contributing to OpenVINO <https://medium.com/openvino-toolkit/how-to-contribute-to-an-ai-open-source-project-c741f48e009e>`__.
- Visit `Intel DevHub Discord server <https://discord.gg/7pVRxUwdWG>`__ to join discussions and talk to OpenVINO developers.