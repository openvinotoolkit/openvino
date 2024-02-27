.. {#long_term_support_policy}

Intel® Distribution of OpenVINO™ toolkit Long-Term Support (LTS) Policy
=============================================================================

The Intel® Distribution of OpenVINO™ toolkit is a tool suite for high-performance, deep
learning. With continuous product improvement and security updates in mind, Intel releases
a new version of the Intel® Distribution of OpenVINO™ toolkit every 3-4 months. This standard
release introduces new features and tools, extends support for additional hardware, libraries,
operating systems and public models, and includes security and stability updates.

In June 2020, Intel introduced the first Long-Term Support (LTS) release of
the Intel® Distribution of OpenVINO™ toolkit. The Intel® Distribution of OpenVINO™ toolkit
LTS release is a stable and reliable release maintained for a longer period of time
than the standard releases. Additionally, the LTS release reduces potential risk and costs
associated with upgrading versions, such as accidentally introducing new bugs and breaking
old functionality. The LTS release is intended for the hardening of the functionality
in existing features as opposed to the introduction of new features.

Sample Use Cases
##################

* **Standard Release (3-4 releases a year):** Users looking to take advantage of new features,
  tools and support in order to keep current with the advancements in deep learning technologies
* **Long-Term Support Release:** Users looking for a stable and reliable version that is
  maintained for a longer period of time, and are looking for little to no new feature changes


Intel® Distribution of OpenVINO ™ toolkit Long-Term Support Releases
########################################################################

* **Trigger Events:** Scenarios, when a new LTS release will be published, are as follows:

  * Critical issues, such as application hang, crash, memory leak, user-specific security issues.
  * An environment update occurs and produces a new issue. An environment update includes:

    * a new operating system (OS) release (e.g. Ubuntu 18.04.3 → 18.04.4).
    * new security patches in the OS kernel, in the build compiler, or others.

  * A Critical or High issue is raised in the `Common Vulnerabilities and Exposures (CVEs)
    database <https://www.cvedetails.com/product/52434/Intel-Openvino.html>`__ , which affects
    one of the third-party components used in the OpenVINO™ toolkit.
  * A security update in a third-party component used inside the OpenVINO™ toolkit, which
    introduces blocking issues for users.
  * Intel introduces new hardware that is supported by the Intel® Distribution of OpenVINO™
    toolkit

    .. note::

       Support for a new hardware instruction set will NOT be provided with LTS releases,
       because it requires significant functional and structural changes that may accidentally
       introduce new issues.

* **Issue Reporting:** To report issues, use the `Intel® Premier Support <https://www.intel.com/content/www/us/en/design/support/ips/training/welcome.html>`__
  clearly stating the issue, impact and expected timeline.
* **Lifecycle:** New LTS releases will be introduced every year. In the first release,
  we guarantee both functional and CVEs fixes backport into the LTS release. For one additional
  year, we will include backport for fixes that are only CVEs as opposed to those that are both
  functional and CVE fixes.
* **Distribution:**

  * `Selector tool <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html>`__
    of all distribution options.
  * Source code distribution: `GitHub <https://github.com/openvinotoolkit/openvino>`__ and
    `Gitee <https://gitee.com/openvinotoolkit-prc/openvino>`__ .
  * Binary distribution:

    * Download from `OpenVINO storage <https://storage.openvinotoolkit.org/repositories/openvino/packages/>`__
    * `pypi.org <https://pypi.org/project/openvino-dev/>`__
    * `DockerHub* <https://hub.docker.com/u/openvino>`__

* **Application Binary Interface (ABI) and release usage:** There is no application binary
  interface (ABI) compatibility for the OpenVINO™ Runtime (Inference Engine). It is recommended
  to re-build your application each time when getting updates.
* **Backward Compatibility** is supported:

  * If you created and compiled your application with the version after the last LTS using
    the OpenVINO API 2.0, your API calls will be working with the current LTS version too.
  * The OpenVINO supports IR version(s) introduced with the LTS release and the IR version(s)
    introduced in the previous release (for example IRv6 for 2020.1; IRv6 and IRv7 for 2020.2).
  * Environment variables and directory structures are frozen for LTS releases, which means
    there will be no structural changes allowed.
  * With the introduction of a new major version in the standard releases, backward compatibility
    may break in the `OpenVINO API 2.0 <https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html>`__
    (e.g. 2022.3 → 2023.0).

* **Forward Compatibility** is NOT supported:

  * If you created and compiled your application with a newer version than the LTS, there
    is no guarantee that you can build your application with the OpenVINO runtime from the LTS
    release.
  * OpenVINO runtime does NOT support IR version(s) introduced with the Model Optimizer
    from a newer version than the LTS release.

Components Included in the Long-Term Support Releases
########################################################

* Components that expose the OpenVINO API 2.0, such as:

  * plugins for OpenVINO runtime, including the following devices: AUTO device, Intel CPU,
    Intel® Processor Graphics (GPU), Intel® Movidius™ VPU (Myriad), Intel® Vision Accelerator
    Design with Intel® Movidius™ VPUs (HDDL), Intel® Gaussian & Neural Accelerator (Intel® GNA).
  * underlying dependencies with low-level libraries - oneAPI Threading Building Blocks (oneTBB),
    oneAPI Deep Neural Network Library (oneDNN), etc.
  * underlying dependencies with hardware-specific OpenCL compilers, drivers and firmware.

* Development tools in the OpenVINO™ toolkit, such as:

  * Model Optimizer,
  * Post-training Optimization Tool,
  * Deep Learning Workbench (DL Workbench).

Functionalities Supported in the Long-Term Support Releases
#################################################################

* Go to the `system requirements <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html>`__
  as well as `official documentation <https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html>`__
  list of supported devices, model formats, input or output precision, input or output layout,
  and supported layers.
* For the best-known configurations, review the documentation on how to
  :doc:`Get a Deep Learning Model Performance Boost with Intel platforms <openvino_docs_performance_benchmarks>`.
* `Release Notes <https://wiki.ith.intel.com/display/CVSDK/Release+Notes+-+OpenVINO+v.2022.3+LTS>`__
  include system requirements (supported hardware targets and corresponding operating systems)

Testing Supported in the Long-Term Support Policy
#####################################################

* No regression allowed: Each user issue must be covered with the corresponding regression test.
* White-box: Unit, behaviour and functional tests
* Black-box: Performance, backward compatibility, load (7x24) and stress testing
* Security: Code coverage, static analysis, BDBA scans, and others.

Components NOT Included in the Long-Term Support Releases
##############################################################

* Fixes for the OpenVINO `deprecated API <https://docs.openvino.ai/nightly/deprecated.html>`__
  will not be introduced.
* The remaining components that are not covered with the LTS release, such as:

  * Open Model Zoo
  * OpenCV
  * Samples
  * Demos

