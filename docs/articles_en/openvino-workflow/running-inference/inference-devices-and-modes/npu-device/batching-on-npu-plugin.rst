NPU Plugin Batching 
===============================


.. meta::
   :description: The Bathing is handled on the NPU plugin in OpenVINO™
                 in two different modes, concurrency-based inferences
                 or handled by the compiler.


The NPU plugin will first check if the following conditions are met:
* Batch size is on the first axis.
* All inputs and outputs have the same batch size.
* Model does not contain states.

In case conditions are met, due to current compiler limitations and ongoing work on performance improvements for batch_size higher than one,
the NPU plugin will first try to compile and execute the original model with forced batch_size to 1.
In case this compilation succeeds, the plugin will detect a difference between the original model layout
and transformed/compiled layout (in batch size) and would:
- internally construct multiple command lists, one for each input
- execute each command list for proper offsets of input/output buffers
- once all command lists are executed, the plugin will notify the user of the completion of the inference request.

This batching mode based on concurrency is transparent to the application. One single inference request will handle all inputs from the batch.
Performance might be lower compared to regular batching; this mode is intended to offer basic batching functionality on older drivers
or in case the model cannot yet be compiled with a batch size larger than one.

In case these conditions are not met the NPU plugin will try to compile and execute the original model with the given
batch_size to N as any other regular model.

Note: Once the performance improves and multiple models can compile with a batch size larger than one,
the default order will be changed; NPU will try first to compile and execute the original model with the given
batch size and fall back to concurrent batching if compilation fails.
