Optimizing for Latency
======================


.. toctree::
   :maxdepth: 1
   :hidden:

   optimizing-latency/model-caching-overview

.. meta::
   :description: OpenVINO provides methods that help to preserve minimal
                 latency despite the number of inference requests and
                 improve throughput without degrading latency.


An application that loads a single model and uses a single input at a time is
a widespread use case in deep learning. Surely, more requests can be created if
needed, for example to support :ref:`asynchronous input population <async_api>`.
However, **the number of parallel requests affects inference performance**
of the application.

Also, running inference of multiple models on the same device relies on whether the models
are executed simultaneously or in a chain: the more inference tasks at once, the higher the
latency.

However, devices such as CPUs and GPUs may be composed of several "sub-devices". OpeVINO can
handle them transparently, when serving multiple clients, improving application's throughput
without impacting latency. What is more, multi-socket CPUs can deliver as many requests at the
same minimal latency as there are NUMA nodes in the system. Similarly, a multi-tile GPU,
which is essentially multiple GPUs in a single package, can deliver a multi-tile
scalability with the number of inference requests, while preserving the
single-tile latency.

.. note::

   Balancing throughput and latency by manual configuration requires strong expertise
   in this area. Instead, you should specify :doc:`performance hints <high-level-performance-hints>`
   for ``compile_model``, which is a device-agnostic and future-proof option.


**For running multiple models simultaneously**, consider using separate devices for each of
them. When multiple models are executed in parallel on a device, use ``ov::hint::model_priority``
to define relative priorities of the models. Note that this feature may not be available for
some devices.

**First-Inference Latency and Model Load/Compile Time**

First-inference latency is the longest time the application requires to finish inference.
This means it includes the time to load and compile the model, which happens at the first
execution only. For some scenarios it may be a significant factor, for example, if the model is
always used just once or is unloaded after each run to free up the memory.

In such cases the device choice is especially important. The CPU offers the fastest model load
time nearly every time. Other accelerators usually take longer to compile a model but may be
better for inference. In such cases, :doc:`Model caching <optimizing-latency/model-caching-overview>`
may reduce latency, as long as there are no additional limitations in write permissions
for the application.

To improve "first-inference latency", you may choose between mapping the model into memory
(the default option) and reading it (the older solution). While mapping is better in most cases,
sometimes it may increase latency, especially when the model is located on a removable or a
network drive. To switch between the two, specify the
`ov::enable_mmap() <../../../api/ie_python_api/_autosummary/openvino.frontend.FrontEnd.html#openvino.frontend.FrontEnd.load>`
property for the ``ov::Core`` as either ``True`` or ``False``.

You can also use :doc:`AUTO device selection inference mode <../inference-devices-and-modes/auto-device-selection>`
to deal with first-inference latency.
It starts inference on the CPU, while waiting for the proper accelerator to load
the model. At that point, it shifts to the new device seamlessly.

.. note::

   Keep in mind that any :doc:`throughput-oriented options <optimizing-throughput>`
   may significantly increase inference time.
