:orphan:

OpenVINO™ Telemetry
=====================


.. meta::
   :description: Learn about OpenVINO™ telemetry, that collects anonymous usage data for the purpose of improving OpenVINO™ tools.


To facilitate debugging and further development, OpenVINO™ collects anonymous telemetry data. Anonymous telemetry data is collected by default,
but you can stop data collection anytime by running the command ``opt_in_out --opt_out``.
It does not extend to any other Intel software, hardware, website usage, or other products.

Google Analytics is used for telemetry purposes. Refer to
`Google Analytics support <https://support.google.com/analytics/answer/6004245#zippy=%2Cour-privacy-policy%2Cgoogle-analytics-cookies-and-identifiers%2Cdata-collected-by-google-analytics%2Cwhat-is-the-data-used-for%2Cdata-access>`__ to understand how the data is collected and processed.

Enable or disable Telemetry reporting
###########################################################

Changing consent decision
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can change your data collection decision with the following command lines:

``opt_in_out --opt_in`` - enable telemetry

``opt_in_out --opt_out`` - disable telemetry


Running without Telemetry
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``openvino-telemetry`` is installed by default with OpenVINO. In environments where sending outbound telemetry is prohibited, network access is restricted, or the filesystem is read-only (such as air-gapped containers), you can disable telemetry using the ``OPENVINO_TELEMETRY_OPTOUT`` environment variable or run OpenVINO without the ``openvino-telemetry`` package installed.

**Using the environment variable**

Set ``OPENVINO_TELEMETRY_OPTOUT=1`` to disable all telemetry collection and prevent writing any local telemetry or consent files:

.. code-block:: sh

   export OPENVINO_TELEMETRY_OPTOUT=1

.. note::

   The ``OPENVINO_TELEMETRY_OPTOUT`` environment variable is supported in ``openvino-telemetry>=2026.5.0``.

**Excluding or removing the package**

Users who wish to run OpenVINO without the telemetry package can exclude or remove it:

- Using pip: ``pip uninstall openvino-telemetry``
- Using uv or lockfile tools:

.. code-block:: toml

   [tool.uv]
   exclude-dependencies = ["openvino-telemetry"]

When ``openvino-telemetry`` is absent, OpenVINO automatically uses an internal silent stub that performs no network requests and generates no local telemetry files.


Telemetry Data Collection Details
###########################################################

.. tab-set::

   .. tab-item:: Telemetry Data Collected
      :sync: telemetry-data-collected

      * Failure reports
      * Error reports
      * Usage data

   .. tab-item:: Tools Collecting Data
      :sync: tools-collecting-data

      * Model conversion API
      * Model Downloader
      * Accuracy Checker
      * Post-Training Optimization Toolkit
      * Neural Network Compression Framework
      * Model Converter
      * Model Quantizer

   .. tab-item:: Telemetry Data Retention
      :sync: telemetry-data-retention

      Telemetry data is retained in Google Analytics for a maximum of 14 months.
      Any raw data that has reached the 14-month threshold is deleted from Google Analytics on a monthly basis.


