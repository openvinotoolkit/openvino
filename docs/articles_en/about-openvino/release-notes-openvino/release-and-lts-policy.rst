.. {#release_long_term_support_policy}

Release and Long-Term Support Policy
=============================================================================

With the release cadence of around one month, OpenVINO receives multiple versions throughout
the yearly cycle. These versions are not continuously supported but rather superseded
by consecutive releases. They should be considered for dynamically-developed solutions,
especially when:

* The newest features and tools are required.
* The solution benefits from frequent changes in supported hardware, libraries,
  operating systems, and models.

The yearly cycle ends with a Long-Term Support (LTS) release, which is not superseded
by the following version. Instead, it continues to be supported, receiving bug fixes
for the first year and security updates for the entire period of two years. The LTS version
is advised for stable production environments, especially when:

* The highest level of security and stability is required.
* The solution does not require frequent changes in hardware or model support.
* The process of upgrading the solution is more demanding, e.g. due to a high complexity
  of the project.


.. note::

   Not all components associated with the OpenVINO toolkit are covered by the LTS policy.
   The following elements are not guaranteed to receive updates:

   * Components in the deprecation period.
   * Components not directly connected to the OpenVINO workflow, such as: Samples, demos,
     and Jupyter notebooks.
   * Code samples used in component testing.


Release Compatibility
########################

**Backward Compatibility is supported:**

* If you create and compile your application with a version newer than the last LTS your API
  calls will be compatible with the LTS version as well.
* OpenVINO supports IR versions introduced in both with the LTS release and the previous release
  (for example IRv6 for 2020.1; IRv6 and IRv7 for 2020.2).
* For an LTS version, no structural changes are allowed, which means that environment variables
  and directory structures are frozen.
* With the introduction of a new major version in the standard releases, backward compatibility
  may break in the OpenVINO API (e.g. 2022.3 → 2023.0).

**Forward Compatibility is NOT supported:**

* If you create and compile your application with a version newer than the most recent LTS,
  there is no guarantee that you can build your application with the OpenVINO runtime from
  the LTS release.
* OpenVINO runtime does **NOT** support OpenVINO IR versions introduced by a version newer than
  the LTS release.

.. note::

   There is no Application Binary Interface (ABI) compatibility for the OpenVINO™ Runtime.
   It is recommended to re-build your application with each update.


**LTS Lifecycle:**

* New LTS releases are published at the end of every year cycle.
* An LTS release receives security updates for the duration of the entire LTS period, which
  is two years (or until superseded by two consecutive LTS versions).
* An LTS release receives updates targeting newly recognized bugs for the period of one year.


**LTS Update Triggers:**

* An LTS version may receive an update when:
* Critical issues are recognized, such as application hang, crash, and memory leak.
* A Critical or High issue is raised in the `Common Vulnerabilities and Exposures (CVEs)
  database <https://www.cvedetails.com/product/52434/Intel-Openvino.html>`__
  that affects one of the third-party components used in the OpenVINO™ toolkit.
* A new issue arises due to an environment update, such as:

  * a new operating system (OS) release (e.g. Ubuntu 18.04.3 → 18.04.4).
  * new security patches in the OS kernel, in the build compiler, or others.

* A security update for one of the third-party components used in the OpenVINO™ toolkit results in new blocking issues for users.
* New hardware is released that is supported by the Intel® Distribution of OpenVINO™ toolkit.
* User-specific security issues are approved for fixing.

.. note::

   Support for new hardware instruction sets will **NOT** be provided for LTS releases, due to
   the risk of introducing new issues.



**LTS Testing**

* **No regression allowed:** each user issue must be covered with the corresponding regression test.
* **White-box:** Unit, behavior, and functional tests.
* **Black-box:** Performance, backward compatibility, load (7x24) and stress testing.
* **Security:** Code coverage, static analysis, BDBA scans, and others.


Additional Resources
########################


**Issue Reporting**

To report issues, use the `Intel® Premier Support <https://www.intel.com/content/www/us/en/design/support/ips/training/welcome.html>`__
clearly stating the issue, impact and expected timeline.


**Distribution:**

* `Selector tool <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html>`__ of all distribution options.
* Source code distribution: `GitHub <https://github.com/openvinotoolkit/openvino>`__ and
  `Gitee <https://gitee.com/openvinotoolkit-prc/openvino>`__ .
* Binary distribution:

  * Download from `OpenVINO storage <https://storage.openvinotoolkit.org/repositories/openvino/packages/>`__
  * `pypi.org <https://pypi.org/project/openvino-dev/>`__
  * `DockerHub* <https://hub.docker.com/u/openvino>`__


