# OpenVINO™ Telemetry {#openvino_docs_telemetry_information}

@sphinxdirective

.. meta::
   :description: Learn about OpenVINO™ telemetry, that with your explicit consent 
                 collects only usage data to simplify debugging and further development.


To facilitate debugging and further development, OpenVINO™ asks its users for 
a permission to collect telemetry data. It will not be collected 
without an explicit consent on your part and will cover only OpenVINO™ usage information.
It does not extend to any other Intel software, hardware, website usage, or other products. 

Google Analytics is used for telemetry purposes. Refer to 
`Google Analytics support <https://support.google.com/analytics/answer/6004245#zippy=%2Cour-privacy-policy%2Cgoogle-analytics-cookies-and-identifiers%2Cdata-collected-by-google-analytics%2Cwhat-is-the-data-used-for%2Cdata-access>`__ to understand how the data is collected and processed.

Enable or disable Telemetry reporting
###########################################################

First-run consent
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

On the first run of an application that collects telemetry data, you will be prompted 
to opt in or out of telemetry collection with the following telemetry message: 

.. code-block:: console

   Intel would like your permission to collect software performance and usage data
   for the purpose of improving Intel products and services. This data will be collected
   directly by Intel or through the use of Google Analytics. This data will be stored 
   in countries where Intel or Google operate.

   You can opt-out at any time in the future by running ``opt_in_out --opt_in``.
   
   More Information is available at docs.openvino.ai.

   Please type ``Y`` to give your consent or ``N`` to decline.

Choose your preference by typing ``Y`` to enable or ``N`` to disable telemetry. Your choice will 
be confirmed by a corresponding disclaimer. If you do not reply to the telemetry message, 
your telemetry data will not be collected. 

For the Neural Network Compression Framework (NNCF), which is not a command line application, 
the telemetry message will not display. Telemetry data will only be collected from NNCF 
if you have explicitly provided consent in another OpenVINO tool.


Changing consent decision
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can change your data collection decision with the following command lines: 

``opt_in_out --opt_in`` - enable telemetry

``opt_in_out --opt_out`` - disable telemetry


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
   
      Telemetry data is retained in Google Analytics for a maximum of 26 months.
      Any raw data that has reached the 26-month threshold is deleted from Google Analytics on a monthly basis.  


@endsphinxdirective