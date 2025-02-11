NPU Plugin Batching 
===============================


.. meta::
   :description: OpenVINO™ NPU plugin supports batching
                 either by executing concurrent inferences or by
                 relying on native compiler support for batching.

OpenVINO™ NPU plugin supports batching either by executing concurrent inferences or by relying on native compiler support for batching.

First, the NPU plugin checks if the following conditions are met:

* The batch size is on the first axis.
* All inputs and outputs have the same batch size.
* The model does not contain states.

**If the conditions are met**, the NPU plugin attempts to compile and execute the original model with batch_size forced to 1. This approach is due to current compiler limitations and ongoing work to improve performance for batch_size greater than one.
If the compilation is successful, the plugin detects a difference in batch size between the original model layout (with a batch size set to N)
and the transformed/compiled layout (with a batch size set to 1). Then it executes the following steps:

1. Internally constructs multiple command lists, one for each input.
2. Executes each command list for the proper offsets of input/output buffers.
3. Notifies the user of the completion of the inference request after all command lists have been executed.

This concurrency-based batching mode is transparent to the application. A single inference request handles all inputs from the batch.
While performance may be lower compared to regular batching (based on native compiler support), this mode provides basic batching functionality for use either with older drivers
or when the model cannot yet be compiled with a batch size larger than one.

**If the conditions are not met**, the NPU plugin tries to compile and execute the original model with the given
batch_size to N as any other regular model.

.. note::

   With future performance improvements and support for compiling multiple models with a batch size larger 
   than one, the default order will change. NPU will try first to compile and execute the original model with the 
   given batch size and fall back to concurrent batching if compilation fails.
