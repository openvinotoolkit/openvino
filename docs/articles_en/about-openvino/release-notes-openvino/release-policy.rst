.. {#release_policy}

Release Policy
=============================================================================

| Monthly release :
|
|    OpenVINO™ is published with a cadence of around one month. Each release is supported only until
|    the next version arrives.
|    This makes regular releases suitable only for:

       * Dynamically developed solutions, benefiting from the newest technologies.
       * Products requiring frequent changes in supported hardware, libraries, operating systems, and models.

| :ref:`Long-Term Support (LTS) <long-term-support-policy>` :
|
|    Each year’s final release becomes a Long-Term Support version, which continues to be supported
|    with bug fixes and security updates, even after newer versions are published.
|    Therefore, LTS is advised for production environments where:

       * The highest level of security and stability is required.
       * There is no need for frequent changes in hardware or model support.
       * Upgrading is challenging, e.g., due to a high software complexity.

| :ref:`Nightly packages <nightly-packages>` :
|
|    OpenVINO also offers nightly packages, as a preview of most recent changes. Due to potential
|    stability and performance issues, these should never be used for production purposes.
|    Instead, they may serve:

       * Early integration testing.
       * Previewing newest features/improvements.

.. _long-term-support-policy:

Long-Term Support Policy
###########################

**LTS Lifecycle**

* New LTS releases are published at the end of every year cycle.
* An LTS release receives security updates for the duration of the entire LTS period, which is two years
  (or until superseded by two consecutive LTS versions).
* An LTS release receives updates targeting newly recognized bugs for the period of one year.

**Components covered by LTS**

* Not all components associated with the OpenVINO™ toolkit are covered by the LTS policy.
  The following elements are not guaranteed to receive updates:
* Components in the deprecation period.
* Components not directly connected to the OpenVINO™ workflow, such as: Samples, demos, and Jupyter notebooks.
* Code samples used in component testing.

**LTS Testing**

* **No regression allowed:** each user issue must be covered with the corresponding regression test.
* **White-box:** Unit, behavior, and functional tests.
* **Black-box:** Performance, backward compatibility, load (7x24) and stress testing.
* **Security:** Code coverage, static analysis, BDBA scans, and others.

.. _nightly-packages:

Nightly Packages
###########################

OpenVINO nightly packages are released every workday.
The following package distributions are available for installation:

* OV Archives on S3
* OV Wheels on S3
* OV Wheels on PyPI (not recommended)

.. tab-set::

   .. tab-item:: OV Archives on S3
      :sync: archives-s3

      1. Go to `OpenVINO Nightly Packages on S3 <https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/>`__.
      2. Select a package you want to install.
      3. Download the archive for your platform.
      4. Unpack the archive in a convenient location.
      5. Once unpacked, proceed as with a regular OpenVINO archive.

   .. tab-item:: OV Wheels on S3
      :sync: wheels-s3

      PyPI repository deployed on AWS S3 (more details `here <https://peps.python.org/pep-0503/>`__ )
      allow usage of a regular PyPI without renaming wheels. Installation commands vary depending
      on the branch:

      .. tab-set::

        .. tab-item:: Master
           :sync: master

           .. code-block:: py

              pip install --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

        .. tab-item:: Release
           :sync: release

           * This command includes **Release Candidates**.
           * To use ``extra-index-url``, you need to pass a link containing ``simple``.
           * The ``–pre`` allows the installation of dev-builds.

           .. code-block:: py

              pip install --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release

   .. tab-item:: OV Wheels on PyPi
      :sync: wheels-pypi

      .. warning::

         Python users should use the **OV Wheels on S3** package.

      Install OV Wheels from PyPI:

      .. code-block:: py

         pip install openvino-nightly

Determing the OpenVINO version
--------------------------------

There are two ways to determine which version of OpenVINO is included in the package:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      Execute the following command within the installed package:

      .. code-block:: python

         python3 -c "import openvino; print(openvino.__version__)"

   .. tab-item:: Archives
      :sync: archives

      You can find the file version in:

      .. code-block::

         <UNZIPPED_ARCHIVE_ROOT>/runtime/version.txt


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


