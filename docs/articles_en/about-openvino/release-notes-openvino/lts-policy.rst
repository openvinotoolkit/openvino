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






