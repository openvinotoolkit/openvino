# Prerelease Information {#prerelease_information}

@sphinxdirective

OpenVINO follows a four-month release cycle, which means three major releases a year, 
the last one being an LTS version. To ensure you do not have to wait long to test its new features, 
OpenVINO developers continue to roll out prerelease versions. In this page you can find
a general changelog and the schedule for all versions for the current year.

.. note:: 
   These versions are pre-release software and have not undergone full validation or qualification. OpenVINO™ toolkit pre-release is:

   * NOT to be incorporated into production software/solutions.
   * NOT subject to official support.
   * Subject to change in the future.
   * Introduced to allow early testing and get early feedback from the community.
 

.. dropdown:: OpenVINO Toolkit 2023.0.0.dev20230217
   :open:
   :animate: fade-in-slide-down
   :color: primary

   OpenVINO™ repository tag: `2023.0.0.dev20230217 <https://github.com/openvinotoolkit/openvino/releases/tag/2023.0.0.dev20230217>`__

   * Enabled PaddlePaddle Framework 2.4
   * Preview of TensorFlow Lite Front End – Load models directly via “read_model” into OpenVINO Runtime and export OpenVINO IR format using Model Optimizer or “convert_model”
   * Introduced new option ov::auto::enable_startup_fallback / ENABLE_STARTUP_FALLBACK to control whether to use CPU to accelerate first inference latency for accelerator HW devices like GPU.
   * New FrontEndManager register_front_end(name, lib_path) interface added, to remove “OV_FRONTEND_PATH” env var (a way to load non-default frontends).


@endsphinxdirective