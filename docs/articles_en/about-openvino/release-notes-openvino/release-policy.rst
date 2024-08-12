Release Policy
=============================================================================

OpenVINO™ offers releases of four different types, each targeting a different use case:

* `Regular releases <#regular-releases>`__
* `Long-Term Support <#long-term-support-releases>`__
* `Pre-release releases <#pre-release-releases>`__
* `Nightly <#nightly-releases>`__

Regular releases
####################

OpenVINO™ is published multiple times a year, when significant new features and bug fixes have
been completed and validated. For each regular release, a dedicated development branch in GitHub
is created, targeting changes such as:

* New features of gold quality, as well as Beta features, labeled as “preview.”
* Key bug fixes.
* Newest hardware support.

Each regular release is supported until the next version arrives, making it suitable for:

* Most typical use cases (the recommended release type).
* Products requiring frequent changes in supported hardware, libraries, operating systems, and models.


Long-Term Support releases
###########################

Each year's final release becomes a Long-Term Support (LTS) version, which continues to receive
bug fixes and security updates, even after newer versions are published. Therefore, LTS may be
used for production environments where:

* There is no need for frequent changes in hardware or model support.
* New optimizations are not prioritized.
* Upgrading is challenging, e.g., due to high software complexity.
* A legacy feature, discontinued in newer OpenVINO versions, is still required.

**LTS Lifecycle**

* LTS is typically published at the end of every year cycle.
* LTS uses the branch of the last yearly regular release.
* LTS aim to receive an update once a year.
* Security updates are offered for the duration of the entire LTS period, which is two years
  (or until superseded by two consecutive LTS versions).
* Updates targeting newly discovered bugs are offered for the period of one year.

.. note::
   LTS releases may offer limited distribution options.

**Components covered by LTS**

Not all components associated with the OpenVINO™ toolkit are covered by the LTS policy.
The following elements are not guaranteed to receive updates:

* Components in the deprecation period.
* Preview features (highlighted in the release notes).
* Components not directly connected to the OpenVINO™ workflow, such as: Samples, demos, and Jupyter notebooks.
* OpenVINO tools, such as NNCF and OVMS.
* Code samples used in component testing.

Pre-release releases
######################

OpenVINO pre-release is an early version of regular releases that has not undergone full release validation
or qualification. Pre-release versions are more stable than nightly releases. No support is offered on pre-release software. The scope, functionality,
and APIs/behavior are subject to change in the future. It **should NOT** be incorporated into
any production software/solution, instead it should be used only for:

* Performing early testing and integration.
* Getting early feedback from the community.

Nightly releases
###########################

OpenVINO nightly releases are the first source of newly added features and priority bug fixes
reported for the previous versions, as a preview of the most recent changes. They are:

* Released every workday.
* Based on the master branch of the OpenVINO GitHub repository.
* Not fit for production environments.
* Offered with limited distribution options:

Since their validation scope is limited, **they should never be used for production purposes**.
Instead, they may serve:

* Early integration testing.
* Community contribution development and integration.
* Tracking development progress.

.. tab-set::

   .. tab-item:: Downloadable Archives
      :sync: archives-s3

      1. Go to `OpenVINO Nightly Packages <https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/>`__.
      2. Select a package you want to install.
      3. Download the archive for your platform.
      4. Unpack the archive in a convenient location.
      5. Once unpacked, proceed as with a regular OpenVINO archive (see :doc:`installation guides <../../../get-started/install-openvino>`).

   .. tab-item:: OV Wheels on S3
      :sync: wheels-s3

      A PyPI repository deployed on AWS S3 (`see more details <https://peps.python.org/pep-0503/>`__)
      enables the use of regular PyPI without the need to rename wheels. Installation commands vary depending
      on the branch:

      .. tab-set::

        .. tab-item:: Master
           :sync: master

           .. code-block:: py

              pip install --pre openvino --extra-index-url
              https://storage.openvinotoolkit.org/simple/wheels/nightly

        .. tab-item:: Release
           :sync: release

           * This command includes **Release Candidates**.
           * To use ``extra-index-url``, you need to pass a link containing ``simple``.
           * The ``--pre`` allows the installation of dev-builds.

           .. code-block:: py

              pip install --pre openvino --extra-index-url
              https://storage.openvinotoolkit.org/simple/wheels/pre-release

   .. tab-item:: OV Wheels on PyPi (not recommended)
      :sync: wheels-pypi


      Install OV Wheels from PyPI:

      .. code-block:: py

         pip install openvino-nightly


Additional Information
#########################

| **Determining the OpenVINO Version**
| If you need to operate on a specific OpenVINO release, and you are not sure which version
  is included in the installed package, you can verify it in one of two ways:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      Execute the following command within the installed package:

      .. code-block:: python

         python3 -c "import openvino; print(openvino.__version__)"

   .. tab-item:: Archives
      :sync: archives

      You can find the file version in:

      .. code-block:: text

         <UNZIPPED_ARCHIVE_ROOT>/runtime/version.txt

| **Issue Reporting**
| To report issues, use the `Intel® Premier Support <https://www.intel.com/content/www/us/en/design/support/ips/training/welcome.html>`__
  clearly stating the issue, impact, and the expected timeline.

| **Distribution:**

* `Selector tool <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html>`__ of all distribution options.
* Source code distribution: `GitHub <https://github.com/openvinotoolkit/openvino>`__ and
  `Gitee <https://gitee.com/openvinotoolkit-prc/openvino>`__ .
* Binary distribution:

  * Download from `OpenVINO storage <https://storage.openvinotoolkit.org/repositories/openvino/packages/>`__
  * `pypi.org <https://pypi.org/project/openvino-dev/>`__
  * `DockerHub* <https://hub.docker.com/u/openvino>`__


