Automatic Device Selection
==========================

.. meta::
   :description: The Automatic Device Selection mode in OpenVINO™ Runtime
                 detects available devices and selects the optimal processing
                 unit for inference automatically.


.. toctree::
   :maxdepth: 1
   :hidden:

   Debugging Auto-Device Plugin <auto-device-selection/debugging-auto-device>


The Automatic Device Selection mode, or AUTO for short, uses a "virtual" or a "proxy" device,
which does not bind to a specific type of hardware, but rather selects the processing unit
for inference automatically. It detects available devices, picks the one best-suited for the
task, and configures its optimization settings. This way, you can write the application once
and deploy it anywhere.

The selection also depends on your performance requirements, defined by the “hints”
configuration API, as well as device priority list limitations, if you choose to exclude
some hardware from the process.

The logic behind the choice is as follows:

1. Check what supported devices are available.
2. Check precisions of the input model (for detailed information on precisions read more on the ``ov::device::capabilities``).
3. Select the highest-priority device capable of supporting the given model, as listed in the table below.
4. If model's precision is FP32 but there is no device capable of supporting it, offload the model to a device supporting FP16.


+----------+-----------------------------------------------------+------------------------------------+
| Device   | Supported                                           | Supported                          |
| Priority | Device                                              | model precision                    |
+==========+=====================================================+====================================+
| 1        | dGPU                                                | FP32, FP16, INT8, BIN              |
|          | (e.g. Intel® Iris® Xe MAX)                          |                                    |
+----------+-----------------------------------------------------+------------------------------------+
| 2        | iGPU                                                | FP32, FP16, BIN                    |
|          | (e.g. Intel® UHD Graphics 620 (iGPU))               |                                    |
+----------+-----------------------------------------------------+------------------------------------+
| 3        | Intel® CPU                                          | FP32, FP16, INT8, BIN              |
|          | (e.g. Intel® Core™ i7-1165G7)                       |                                    |
+----------+-----------------------------------------------------+------------------------------------+
| 4        | Intel® NPU                                          |                                    |
|          | (e.g. Intel® Core™ Ultra)                           |                                    |
+----------+-----------------------------------------------------+------------------------------------+

.. note::

   Note that NPU is currently excluded from the default priority list. To use it for inference, you
   need to specify it explicitly


How AUTO Works
##############

To put it simply, when loading the model to the first device on the list fails, AUTO will try to load it to the next device in line, until one of them succeeds.
What is important, **AUTO starts inference with the CPU of the system by default unless there is model cached for the best suited device**, as it provides very low latency and can start inference with no additional delays.
While the CPU is performing inference, AUTO continues to load the model to the device best suited for the purpose and transfers the task to it when ready.
This way, the devices which are much slower in compiling models, GPU being the best example, do not impact inference at its initial stages.
For example, if you use a CPU and a GPU, the first-inference latency of AUTO will be better than that of using GPU alone.

Note that if you choose to exclude CPU from the priority list or disable the initial
CPU acceleration feature via ``ov::intel_auto::enable_startup_fallback``, it will be
unable to support the initial model compilation stage. The models with :doc:`stateful operations <../stateful-models>`
will be loaded to the CPU if it is in the candidate list. Otherwise,
these models will follow the normal flow and be loaded to the device based on priority.

.. image:: ../../../assets/images/autoplugin_accelerate.svg


This mechanism can be easily observed in the :ref:`Using AUTO with Benchmark app sample <using-auto-with-openvino-samples-and-benchmark-app>`
section, showing how the first-inference latency (the time it takes to compile the
model and perform the first inference) is reduced when using AUTO. For example:


.. code-block:: sh

   benchmark_app -m ../public/alexnet/FP32/alexnet.xml -d GPU -niter 128


.. code-block:: sh

   benchmark_app -m ../public/alexnet/FP32/alexnet.xml -d AUTO -niter 128



.. note::

   The longer the process runs, the closer realtime performance will be to that of the best-suited device.

.. note::

   **Testing accuracy with the AUTO device is not recommended.** Since the CPU and GPU (or other target devices) may produce slightly different accuracy numbers, using AUTO could lead to inconsistent accuracy results from run to run due to a different number of inferences on CPU and GPU. This is particularly true when testing with a small number of inputs. To achieve consistent inference on the GPU (or another target device), you can disable CPU acceleration by setting ``ov::intel_auto::enable_startup_fallback`` to false.


Using AUTO
##########

Following the OpenVINO™ naming convention, the Automatic Device Selection mode is assigned the label of "AUTO".
It may be defined with no additional parameters, resulting in defaults being used, or configured further with
the following setup options:

+----------------------------------------------+--------------------------------------------------------------------+
| Property(C++ version)                        | Values and Description                                             |
+==============================================+====================================================================+
| <device candidate list>                      | **Values**:                                                        |
|                                              |                                                                    |
|                                              | empty                                                              |
|                                              |                                                                    |
|                                              | ``AUTO``                                                           |
|                                              |                                                                    |
|                                              | ``AUTO: <device names>`` (comma-separated, no spaces)              |
|                                              |                                                                    |
|                                              | Lists the devices available for selection.                         |
|                                              | The device sequence will be taken as priority from high to low.    |
|                                              | If not specified, ``AUTO`` will be used as default,                |
|                                              | and all devices will be "viewed" as candidates.                    |
+----------------------------------------------+--------------------------------------------------------------------+
| ``ov::device::priorities``                   | **Values**:                                                        |
|                                              |                                                                    |
|                                              | ``<device names>`` (comma-separated, no spaces)                    |
|                                              |                                                                    |
|                                              | Specifies the devices for AUTO to select.                          |
|                                              | The device sequence will be taken as priority from high to low.    |
|                                              | This configuration is optional.                                    |
+----------------------------------------------+--------------------------------------------------------------------+
| ``ov::hint::performance_mode``               | **Values**:                                                        |
|                                              |                                                                    |
|                                              | ``ov::hint::PerformanceMode::LATENCY``                             |
|                                              |                                                                    |
|                                              | ``ov::hint::PerformanceMode::THROUGHPUT``                          |
|                                              |                                                                    |
|                                              | ``ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT``               |
|                                              |                                                                    |
|                                              | Specifies the performance option preferred by the application.     |
+----------------------------------------------+--------------------------------------------------------------------+
| ``ov::hint::model_priority``                 | **Values**:                                                        |
|                                              |                                                                    |
|                                              | ``ov::hint::Priority::HIGH``                                       |
|                                              |                                                                    |
|                                              | ``ov::hint::Priority::MEDIUM``                                     |
|                                              |                                                                    |
|                                              | ``ov::hint::Priority::LOW``                                        |
|                                              |                                                                    |
|                                              | Indicates the priority for a model.                                |
|                                              |                                                                    |
|                                              | IMPORTANT: This property is not fully supported yet.               |
+----------------------------------------------+--------------------------------------------------------------------+
| ``ov::execution_devices``                    | Lists the runtime target devices on which the inferences are being |
|                                              | executed.                                                          |
|                                              |                                                                    |
|                                              | Examples of returning results could be ``(CPU)``(``CPU`` is a      |
|                                              | temporary device, indicating that CPU is used for acceleration at  |
|                                              | the model compilation stage), ``CPU``, ``GPU``, ``CPU GPU``,       |
|                                              | ``GPU.0``, etc.                                                    |
+----------------------------------------------+--------------------------------------------------------------------+
| ``ov::intel_auto::enable_startup_fallback``  | **Values**:                                                        |
|                                              |                                                                    |
|                                              | ``true``                                                           |
|                                              |                                                                    |
|                                              | ``false``                                                          |
|                                              |                                                                    |
|                                              | Enables/disables CPU as acceleration (or the helper device) in the |
|                                              | beginning. The default value is ``true``, indicating that CPU is   |
|                                              | used as acceleration by default.                                   |
+----------------------------------------------+--------------------------------------------------------------------+
| ``ov::intel_auto::enable_runtime_fallback``  | **Values**:                                                        |
|                                              |                                                                    |
|                                              | ``true``                                                           |
|                                              |                                                                    |
|                                              | ``false``                                                          |
|                                              |                                                                    |
|                                              | Enables/disables runtime fallback to other devices and performs    |
|                                              | the failed inference request again, if inference request fails on  |
|                                              | the currently selected device.                                     |
|                                              |                                                                    |
|                                              | The default value is ``true``.                                     |
+----------------------------------------------+--------------------------------------------------------------------+
| ``ov::intel_auto::schedule_policy``          | **Values**:                                                        |
|                                              |                                                                    |
|                                              | ``ROUND_ROBIN``                                                    |
|                                              |                                                                    |
|                                              | ``DEVICE_PRIORITY``                                                |
|                                              |                                                                    |
|                                              | Specify the schedule policy of infer request assigned to hardware  |
|                                              | plugin for AUTO cumulative mode.                                   |
|                                              |                                                                    |
|                                              | The default value is ``DEVICE_PRIORITY``.                          |
+----------------------------------------------+--------------------------------------------------------------------+

Inference with AUTO is configured similarly to when device plugins are used:
you compile the model on the plugin with configuration and execute inference.

The code samples on this page assume following import(Python)/using (C++) are included at the beginning of code snippets.

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto.py
           :language: python
           :fragment: [py_ov_property_import_header]

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/AUTO0.cpp
            :language: cpp
            :fragment: [py_ov_property_import_header]



Device Candidates and Priority
++++++++++++++++++++++++++++++

The device candidate list enables you to customize the priority and limit the choice of devices available to AUTO.

* If <device candidate list> is not specified, AUTO assumes all the devices present in the system can be used.
* If ``AUTO`` without any device names is specified, AUTO assumes all the devices present in the system can be used, and will load the network to all devices and run inference based on their default priorities, from high to low.

To specify the priority of devices, enter the device names in the priority order (from high to low) in ``AUTO: <device names>``, or use the ``ov::device::priorities`` property.

See the following code for using AUTO and specifying devices:

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto.py
           :language: python
           :fragment: [part0]

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/AUTO0.cpp
            :language: cpp
            :fragment: [part0]



Note that OpenVINO Runtime lets you use "GPU" as an alias for "GPU.0" in function calls.
More details on enumerating devices can be found in :doc:`Inference Devices and Modes <../inference-devices-and-modes>`.

Checking Available Devices
--------------------------

To check what devices are present in the system, you can use Device API, as listed below. For information on how to use it, see :doc:`Query device properties and configuration <query-device-properties>`.

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. code-block:: sh

           openvino.runtime.Core.available_devices

        See the Hello Query Device Python Sample for reference.

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: sh

           ov::runtime::Core::get_available_devices()

        See the Hello Query Device C++ Sample for reference.


Excluding Devices from Device Candidate List
--------------------------------------------

You can also exclude hardware devices from AUTO, for example, to reserve CPU for other jobs. AUTO will not use the device for inference then. To do that, add a minus sign ``(-)`` before CPU in ``AUTO: <device names>``, as in the following example:

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. code-block:: sh

           compiled_model = core.compile_model(model=model, device_name="AUTO:-CPU")

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: sh

           ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:-CPU");


AUTO will then query all available devices and remove CPU from the candidate list.

Note that if you choose to exclude CPU from device candidate list, CPU will not be able to
support the initial model compilation stage. See more information in
`How AUTO Works <#auto-device-selection>`__.


Performance Hints for AUTO
++++++++++++++++++++++++++

The ``ov::hint::performance_mode`` property enables you to specify a performance option for AUTO to be more efficient for particular use cases. The default hint for AUTO is ``LATENCY``.

The THROUGHPUT and CUMULATIVE_THROUGHPUT hints below only improve performance in an
asynchronous inference pipeline. For information on asynchronous inference, see the
:doc:`Async API documentation <../integrate-openvino-with-your-application/inference-request>` .
The following notebooks provide examples of how to set up an asynchronous pipeline:

* :doc:`Image Classification Async Sample <../../../learn-openvino/openvino-samples/image-classification-async>`
* `Notebook - Asynchronous Inference with OpenVINO™ <./../../../notebooks/async-api-with-output.html>`__
* `Notebook - Automatic Device Selection with OpenVINO <./../../../notebooks/auto-device-with-output.html>`__

LATENCY
--------------------

This option prioritizes low latency, providing short response time for each inference job. It performs best for tasks where inference is required for a single input image, e.g. a medical analysis of an ultrasound scan image. It also fits the tasks of real-time or nearly real-time applications, such as an industrial robot's response to actions in its environment or obstacle avoidance for autonomous vehicles.

.. note::

   If no performance hint is set explicitly, AUTO will set LATENCY for devices that have not set ``ov::device::properties``, for example, ``ov::device::properties(<DEVICE_NAME>, ov::hint::performance_mode(ov::hint::LATENCY))``.

.. _cumulative throughput:

``THROUGHPUT``
--------------------

This option prioritizes high throughput, balancing between latency and power. It is best suited for tasks involving multiple jobs, such as inference of video feeds or large numbers of images.

``CUMULATIVE_THROUGHPUT``
---------------------------------

While ``LATENCY`` and ``THROUGHPUT`` can select one target device with your preferred performance option,
the ``CUMULATIVE_THROUGHPUT`` option enables running inference on multiple devices for higher throughput.
With ``CUMULATIVE_THROUGHPUT``, AUTO loads the network model to all available devices (specified by AUTO)
in the candidate list, and then runs inference on them based on the default or specified priority.

If device priority is specified when using ``CUMULATIVE_THROUGHPUT``, AUTO will run inference
requests on devices based on the priority. In the following example, AUTO will always
try to use GPU first, and then use CPU if GPU is busy:

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. code-block:: sh

           compiled_model = core.compile_model(model, "AUTO:GPU,CPU", {hints.performance_mode: hints.PerformanceMode.CUMULATIVE_THROUGHPUT})

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: sh

           ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:GPU,CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));


If AUTO is used without specifying any device names, and if there are multiple GPUs in the system, ``CUMULATIVE_THROUGHPUT`` mode will use all of the GPUs by default. If the system has more than two GPU devices, AUTO will remove CPU from the device candidate list to keep the GPUs running at full capacity. A full list of system devices and their unique identifiers can be queried using ov::Core::get_available_devices (for more information, see :doc:`Query Device Properties <query-device-properties>`). To explicitly specify which GPUs to use, set their priority when compiling with AUTO:

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. code-block:: sh

           compiled_model = core.compile_model(model, "AUTO:GPU.1,GPU.0", {hints.performance_mode: hints.PerformanceMode.CUMULATIVE_THROUGHPUT})

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: sh

           ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:GPU.1,GPU.0", ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));


Code Examples
--------------------

To enable performance hints for your application, use the following code:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto.py
           :language: python
           :fragment: [part3]

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/AUTO3.cpp
            :language: cpp
            :fragment: [part3]


Disabling Auto-Batching for THROUGHPUT and CUMULATIVE_THROUGHPUT
----------------------------------------------------------------

The ``ov::hint::PerformanceMode::THROUGHPUT`` mode and the ``ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT`` mode will trigger Auto-Batching (for example, for the GPU device) by default. You can disable it by setting ``ov::hint::allow_auto_batching(false)``, or change the default timeout value to a large number, e.g. ``ov::auto_batch_timeout(1000)``. See :doc:`Automatic Batching <automatic-batching>` for more details.


Configuring Model Priority
++++++++++++++++++++++++++

The ``ov::hint::model_priority`` property enables you to control the priorities of models in the Auto-Device plugin. A high-priority model will be loaded to a supported high-priority device. A lower-priority model will not be loaded to a device that is occupied by a higher-priority model.

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto.py
           :language: python
           :fragment: [part4]

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/AUTO4.cpp
            :language: cpp
            :fragment: [part4]


Checking Target Runtime Devices
+++++++++++++++++++++++++++++++

To query the runtime target devices on which the inferences are being executed using AUTO, you can use the ``ov::execution_devices`` property. It must be used with ``get_property``, for example:


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto.py
           :language: python
           :fragment: [part7]

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/AUTO7.cpp
            :language: cpp
            :fragment: [part7]


Configuring Individual Devices and Creating the Auto-Device plugin on Top
#########################################################################

Although the methods described above are currently the preferred way to execute inference with AUTO, the following steps can be also used as an alternative. It is currently available as a legacy feature and used if AUTO is incapable of utilizing the Performance Hints option.

.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_auto.py
           :language: python
           :fragment: [part5]

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/AUTO5.cpp
            :language: cpp
            :fragment: [part5]


.. _using-auto-with-openvino-samples-and-benchmark-app:

Using AUTO with OpenVINO Samples and Benchmark app
##################################################

To see how the Auto-Device plugin is used in practice and test its performance, take a look at OpenVINO™ samples. All samples supporting the "-d" command-line option (which stands for "device") will accept the plugin out-of-the-box. The Benchmark Application will be a perfect place to start – it presents the optimal performance of the plugin without the need for additional settings, like the number of requests or CPU threads. To evaluate the AUTO performance, you can use the following commands:

For unlimited device choice:

.. code-block:: sh

   benchmark_app –d AUTO –m <model> -i <input> -niter 1000

For limited device choice:

.. code-block:: sh

   benchmark_app –d AUTO:CPU,GPU –m <model> -i <input> -niter 1000

For more information, refer to the :doc:`Benchmark Tool <../../../learn-openvino/openvino-samples/benchmark-tool>` article.

.. note::

   The default CPU stream is 1 if using “-d AUTO”.

   You can use the FP16 IR to work with auto-device.

   No demos are yet fully optimized for AUTO, by means of selecting the most suitable device, using the GPU streams/throttling, and so on.


Additional Resources
####################

* `Automatic Device Selection with OpenVINO™ Notebook <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/auto-device/auto-device.ipynb>`__
* :doc:`Debugging AUTO <auto-device-selection/debugging-auto-device>`
* :doc:`(LEGACY) Running on Multiple Devices Simultaneously <../../../documentation/legacy-features/multi-device>`
* :doc:`Inference Devices and Modes <../inference-devices-and-modes>`


