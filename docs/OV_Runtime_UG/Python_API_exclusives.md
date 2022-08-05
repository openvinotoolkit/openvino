# OpenVINO™ Python API Exclusives {#openvino_docs_OV_UG_Python_API_exclusives}

OpenVINO™ Runtime Python API offers additional features and helpers to enhance user experience. The main goal of Python API is to provide user-friendly and simple yet powerful tool for Python users.

## Easier Model Compilation 

`CompiledModel` can be easily created with the helper method. It hides the creation of `Core` and applies `AUTO` inference mode by default.

@snippet docs/snippets/ov_python_exclusives.py auto_compilation

## Model/CompiledModel Inputs and Outputs

Besides functions aligned to C++ API, some of them have their Python counterparts or extensions. For example, `Model` and `CompiledModel` inputs/outputs can be accessed via properties.

@snippet docs/snippets/ov_python_exclusives.py properties_example

Refer to Python API documentation on which helper functions or properties are available for different classes.

## Working with Tensor

Python API allows passing data as tensors. The `Tensor` object holds a copy of the data from the given array. The `dtype` of *numpy* arrays is converted to OpenVINO™ types automatically.

@snippet docs/snippets/ov_python_exclusives.py tensor_basics

### Shared Memory Mode

`Tensor` objects can share the memory with *numpy* arrays. By specifying the `shared_memory` argument, the `Tensor` object does not copy data. Instead, it has access to the memory of the *numpy* array.

@snippet docs/snippets/ov_python_exclusives.py tensor_shared_mode

### Slices of array's memory

One of the `Tensor` class constructors allows to share the slice of array's memory. When `shape` is specified in the constructor that has the numpy array as first argument, it triggers the special shared memory mode.

@snippet docs/snippets/ov_python_exclusives.py tensor_slice_mode

## Running inference

Python API supports extra calling methods to synchronous and asynchronous modes for inference.

All infer methods allow users to pass data as popular *numpy* arrays, gathered in either Python dicts or lists.

@snippet docs/snippets/ov_python_exclusives.py passing_numpy_array

Results from inference can be obtained in various ways:

@snippet docs/snippets/ov_python_exclusives.py getting_results

### Synchronous Mode - Extended

Python API provides different synchronous calls to infer model, which block the application execution. Additionally, these calls return results of inference:

@snippet docs/snippets/ov_python_exclusives.py sync_infer

### AsyncInferQueue

Asynchronous mode pipelines can be supported with a wrapper class called `AsyncInferQueue`. This class automatically spawns the pool of `InferRequest` objects (also called "jobs") and provides synchronization mechanisms to control the flow of the pipeline.

Each job is distinguishable by a unique `id`, which is in the range from 0 up to the number of jobs specified in the `AsyncInferQueue` constructor.

The `start_async` function call is not required to be synchronized - it waits for any available job if the queue is busy/overloaded. Every `AsyncInferQueue` code block should end with the `wait_all` function which provides the "global" synchronization of all jobs in the pool and ensure that access to them is safe.

@snippet docs/snippets/ov_python_exclusives.py asyncinferqueue

#### Acquiring Results from Requests

After the call to `wait_all`, jobs and their data can be safely accessed. Acquiring a specific job with `[id]` will return the `InferRequest` object, which will result in seamless retrieval of the output data.

@snippet docs/snippets/ov_python_exclusives.py asyncinferqueue_access

#### Setting Callbacks

Another feature of `AsyncInferQueue` is the ability to set callbacks. When callback is set, any job that ends inference calls upon the Python function. The callback function must have two arguments: one is the request that calls the callback, which provides the `InferRequest` API; the other is called "userdata", which provides the possibility of passing runtime values. Those values can be of any Python type and later used within the callback function.

The callback of `AsyncInferQueue` is uniform for every job. When executed, GIL is acquired to ensure safety of data manipulation inside the function.

@snippet docs/snippets/ov_python_exclusives.py asyncinferqueue_set_callback
