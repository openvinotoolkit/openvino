# Debugging Auto-Device Plugin {#openvino_docs_OV_UG_supported_plugins_AUTO_debugging}

## Using Debug Log
In case of execution problems, just like all other plugins, Auto-Device provides the user with information on exceptions and error values. If the returned data is not enough for debugging purposes, more information may be acquired by means of `ov::log::Level`.

There are six levels of logs, which can be called explicitly or set via the `OPENVINO_LOG_LEVEL` environment variable (can be overwritten by `compile_model()` or `set_property()`):

0 - ov::log::Level::NO  
1 - ov::log::Level::ERR  
2 - ov::log::Level::WARNING  
3 - ov::log::Level::INFO  
4 - ov::log::Level::DEBUG  
5 - ov::log::Level::TRACE  

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/AUTO6.cpp
       :language: cpp
       :fragment: [part6]
 
.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto.py
       :language: python
       :fragment: [part6]

.. tab:: OS environment variable

   .. code-block:: sh

      When defining it via the variable, 
      a number needs to be used instead of a log level name, e.g.:
      
      Linux
      export OPENVINO_LOG_LEVEL=0
      
      Windows
      set OPENVINO_LOG_LEVEL=0
@endsphinxdirective

The property returns information in the following format:

@sphinxdirective
.. code-block:: sh

   [time]LOG_LEVEL[file] [PLUGIN]: message
@endsphinxdirective

in which the `LOG_LEVEL` is represented by the first letter of its name (ERROR being an exception and using its full name). For example:

@sphinxdirective
.. code-block:: sh

   [17:09:36.6188]D[plugin.cpp:167] deviceName:GPU, defaultDeviceID:, uniqueName:GPU_
   [17:09:36.6242]I[executable_network.cpp:181] [AUTOPLUGIN]:select device:GPU
   [17:09:36.6809]ERROR[executable_network.cpp:384] [AUTOPLUGIN] load failed, GPU:[ GENERAL_ERROR ]
@endsphinxdirective


## Instrumentation and Tracing Technology

All major performance calls of both OpenVINO™ Runtime and the AUTO plugin are instrumented with Instrumentation and Tracing Technology (ITT) APIs. To enable ITT in OpenVINO™ Runtime, compile it with the following option:
@sphinxdirective
.. code-block:: sh

   -DENABLE_PROFILING_ITT=ON
@endsphinxdirective

For more information, you can refer to:
* [Intel® VTune™ Profiler User Guide](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/api-support/instrumentation-and-tracing-technology-apis.html)

### Analyze Code Performance on Linux

You can analyze code performance using Intel® VTune™ Profiler. For more information and installation instructions refer to the [installation guide (PDF)](https://software.intel.com/content/www/us/en/develop/download/intel-vtune-install-guide-linux-os.html)
With Intel® VTune™ Profiler installed you can configure your analysis with the following steps:

1. Open Intel® VTune™ Profiler GUI on the host machine with the following command:
@sphinxdirective

.. code-block:: sh

   cd /vtune install dir/intel/oneapi/vtune/2021.6.0/env
   source vars.sh
   vtune-gui
@endsphinxdirective

2. select **Configure Analysis**
3. In the **where** pane, select **Local Host**
@sphinxdirective
.. image:: _static/images/OV_UG_supported_plugins_AUTO_debugging-img01-localhost.png
   :align: center
@endsphinxdirective
4. In the **what** pane, specify your target application/script on the local system.
@sphinxdirective
.. image:: _static/images/OV_UG_supported_plugins_AUTO_debugging-img02-launch.png
   :align: center
@endsphinxdirective
5. In the **how** pane, choose and configure the analysis type you want to perform, for example, **Hotspots Analysis**:
identify the most time-consuming functions and drill down to see time spent on each line of source code. Focus optimization efforts on hot code for the greatest performance impact.
@sphinxdirective
.. image:: _static/images/OV_UG_supported_plugins_AUTO_debugging-img03-hotspots.png
   :align: center
@endsphinxdirective
6.	Start the analysis by clicking the start button. When it is done, you will get a summary of the run, including top hotspots and top tasks in your application:
@sphinxdirective
.. image:: _static/images/OV_UG_supported_plugins_AUTO_debugging-img04-vtunesummary.png
   :align: center
@endsphinxdirective
7. To analyze ITT info related to the Auto plugin, click on the **Bottom-up** tab, choose the **Task Domain/Task Type/Function/Call Stack** from the dropdown list - Auto plugin-related ITT info is under the MULTIPlugin task  domain:
@sphinxdirective
.. image:: _static/images/OV_UG_supported_plugins_AUTO_debugging-img05-vtunebottomup.png
   :align: center
@endsphinxdirective
