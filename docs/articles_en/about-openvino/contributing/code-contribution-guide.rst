Code Contribution Guide
=======================

This section will start you off with a few simple steps to begin your code contribution.
If you have any doubts, talk to
`the development team <https://github.com/orgs/openvinotoolkit/teams/openvino-developers/teams>`__.
Remember, your questions help us keep improving OpenVINO.


1. **Choose the issue you want to work on.**

   Choose one of the existing `issues <https://github.com/openvinotoolkit/openvino/issues>`__ /
   requests. The `“Good First Issue” <https://github.com/orgs/openvinotoolkit/projects/3>`__
   board is a good place to start. If you have a new idea for the contribution,
   make sure to first create a proper issue, discussion, or feature request.

   Here are some of the components you may choose to work on.

   .. tab-set::

      .. tab-item:: APIs

         - `Core C++ API <https://github.com/openvinotoolkit/openvino/tree/master/src/core>`__
         - `C API <https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/c>`__
         - `Python API <https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/python>`__
         - `JavaScript (Node.js) API <https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/js>`__

      .. tab-item:: Frontends

         - `IR Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/ir>`__
         - `ONNX Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/onnx>`__
         - `PaddlePaddle Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle>`__
         - `PyTorch Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/pytorch>`__
         - `TensorFlow Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/tensorflow>`__
         - `TensorFlow Lite Frontend <https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/tensorflow_lite>`__

      .. tab-item:: Plugins

         - `Auto plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/auto>`__
         - `CPU plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu>`__
         - `GPU plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu>`__
         - `NPU plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu>`__
         - `Hetero plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/hetero>`__
         - `Template plugin <https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/template>`__

      .. tab-item:: Tools

         - `Benchmark Tool <https://github.com/openvinotoolkit/openvino/tree/master/tools/benchmark_tool>`__
         - `Model Conversion <https://github.com/openvinotoolkit/openvino/tree/master/tools/ovc>`__

2. **Assign yourself to the issue.**

   To get assigned to a task, simply leave a comment with the ``.take`` command in
   the selected issue. You can always ask OpenVINO developers for guidance,
   both technical and organizational:

   - assign users in the **“Contact points”** section,
   - visit `Intel DevHub Discord server <https://discord.gg/7pVRxUwdWG>`__ to ask
     questions in the channel dedicated to **“Good First Issue”** support, or any other.

3. **Build OpenVINO.**

   In order to build OpenVINO, follow the
   `build instructions for your specific OS <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__.

   Use the local build and the information found in the issue description to
   develop your contribution.

4. **Submit a PR with your changes.**

   Follow the `guidelines <https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_PR.md>`__
   and do not forget to `link your Pull Request to the issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#manually-linking-a-pull-request-to-an-issue-using-the-pull-request-sidebar>`__
   it addresses.

5. **Wait for a review.**

   We will make sure to review your **Pull Request** as soon as possible and provide feedback.
   You can expect a merge once your changes have been validated with automatic tests and
   approved by `maintainers <https://github.com/orgs/openvinotoolkit/teams/openvino-maintainers/teams>`__.


Additional Resources
#####################

- Choose a `“Good First Issue” <https://github.com/orgs/openvinotoolkit/projects/3>`__.
- Learn more about `OpenVINO architecture <https://github.com/openvinotoolkit/openvino/blob/master/src/docs/architecture.md>`__.
- Check out a `blog post on contributing to OpenVINO <https://medium.com/openvino-toolkit/how-to-contribute-to-an-ai-open-source-project-c741f48e009e>`__.
- Visit `Intel DevHub Discord server <https://discord.gg/7pVRxUwdWG>`__ to join discussions and talk to OpenVINO developers.