# OpenVINO™ Python API exclusives {#openvino_docs_OV_Runtime_UG_Python_API_exclusives}

OpenVINO™ Runtime Python API is exposing additional features and helpers to elevate user experience. Main goal of Python API is to provide user-friendly and simple, yet powerful, tool for Python users.

## Easier model compilation 

`CompiledModel` can be easily created with helper method. It hides `Core` creation and applies `AUTO` device by default.

@sphinxdirective

.. doxygensnippet:: docs/snippets/ov_python_exclusives.py
    :language: python
    :fragment: [auto_compilation]

@endsphinxdirective

## Model/CompiledModel inputs and outputs

Besides functions aligned to C++ API, some of them have their Pythonic counterparts or extensions. For the example, `Model` and `CompiledModel` inputs/outputs can be accessed via properties.

@sphinxdirective

.. doxygensnippet:: docs/snippets/ov_python_exclusives.py
    :language: python
    :fragment: [properties_example]

@endsphinxdirective

Please refer to Python API documentation on which helper functions or properties are available for different classes.

## Working with Tensor

Python API allows users to pass data as tensors. `Tensor` object holds a copy of the data from the given array. `dtype` of numpy arrays is converted to OpenVINO™ types automatically.

@sphinxdirective

.. doxygensnippet:: docs/snippets/ov_python_exclusives.py
    :language: python
    :fragment: [tensor_basics]

@endsphinxdirective

### Shared memory mode

`Tensor` objects can share the memory with numpy arrays. By specifing `shared_memory` argument, a `Tensor` object does not perform copy of data and has access to the memory of the numpy array.

@sphinxdirective

.. doxygensnippet:: docs/snippets/ov_python_exclusives.py
    :language: python
    :fragment: [tensor_shared_mode]

@endsphinxdirective

### Slices of array's memory

One of the `Tensor` class constructors allows to share the slice of array's memory. When `shape` is specified in the constructor that has the numpy array as first argument, it triggers the special shared memory mode.

@sphinxdirective

.. doxygensnippet:: docs/snippets/ov_python_exclusives.py
    :language: python
    :fragment: [tensor_slice_mode]

@endsphinxdirective

## Running inference

Python API supports extra calling methods to synchronous and asynchronous modes for inference.

All infer methods allow users to pass data as popular numpy arrays, gathered in either Python dicts or lists.

@sphinxdirective

.. doxygensnippet:: docs/snippets/ov_python_exclusives.py
    :language: python
    :fragment: [passing_numpy_array]

@endsphinxdirective

### Synchronous mode - extended

Python API provides different synchronous calls to infer model, which block the application execution, calls return results of inference:

@sphinxdirective

.. doxygensnippet:: docs/snippets/ov_python_exclusives.py
    :language: python
    :fragment: [sync_infer]

@endsphinxdirective

### AsyncInferQueue

Asynchronous mode pipelines can be supported with wrapper class called `AsyncInferQueue`. This class automatically spawns pool of `InferRequest` objects (also called "jobs") and provides synchronization mechanisms to control flow of the pipeline.

Each job is distinguishable by unique `id`, which is in range of 0 up to number of jobs specified in `AsyncInferQueue` constructor.

Function call `start_async` is not required to be synchronized, it waits for any available job if queue is busy/overloaded. Every `AsyncInferQueue` code block should end with `wait_all` function. It provides "global" synchronization of all jobs in the pool and ensure that access to them is safe.

@sphinxdirective

.. doxygensnippet:: docs/snippets/ov_python_exclusives.py
    :language: python
    :fragment: [asyncinferqueue]

@endsphinxdirective

#### Acquire results from requests

After the call to `wait_all`, jobs and their data can be safely accessed. Acquring of a specific job with `[id]` returns `InferRequest` object, which results in seamless retrieval of the output data.

@sphinxdirective

.. doxygensnippet:: docs/snippets/ov_python_exclusives.py
    :language: python
    :fragment: [asyncinferqueue_access]

@endsphinxdirective

#### Setting callbacks

Another feature of `AsyncInferQueue` is ability of setting callbacks. When callback is set, any job that ends inference, calls upon Python function. Callback function must have two arguments. First is the request that calls the callback, it provides `InferRequest` API. Second one being called "userdata", passed values can be of any Python type and later used inside callback function.

The callback of `AsyncInferQueue` is uniform for every job. When executed, GIL is acquired to ensure safety of data manipulation inside the function.

@sphinxdirective

.. doxygensnippet:: docs/snippets/ov_python_exclusives.py
    :language: python
    :fragment: [asyncinferqueue_set_callback]

@endsphinxdirective
