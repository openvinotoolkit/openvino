# High-level Performance Hints {#openvino_docs_OV_UG_Performance_Hints}



## (Automatic) Batching Execution

`ov::hint::PerformanceMode::THROUGHPUT` is specified for the `ov::hint::performance_mode` property for the compile_model or set_property calls.
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [compile_model]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [compile_model]

@endsphinxdirective

Using the hints assumes that the application queries the `ov::optimal_number_of_infer_requests` to create and run the returned number of requests simultaneously:
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [query_optimal_num_requests]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [query_optimal_num_requests]

@endsphinxdirective

For example, the application processes only 4 video streams, so there is no need to use a batch larger than 4. The most future-proof way to communicate the limitations on the parallelism is to equip the performance hint with the optional `ov::hint::num_requests` configuration key set to 4. For the GPU this will limit the batch size, for the CPU - the number of inference streams, so each device uses the `ov::hint::num_requests` while converting the hint to the actual device configuration options:
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [hint_num_requests]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: hint_num_requests]

@endsphinxdirective

### Performance Hints


### See Also
[Supported Devices](supported_plugins/Supported_Devices.md)