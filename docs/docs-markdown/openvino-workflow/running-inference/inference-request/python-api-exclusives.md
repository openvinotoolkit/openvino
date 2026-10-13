---
sidebar_position: 2
sidebar_label: "OpenVINO™ Python API Exclusives"
description: "OpenVINO™ Runtime Python API includes additional features to improve user experience and provide simple yet powerful tool for Python users."
---

# OpenVINO™ Python API Exclusives

OpenVINO™ Runtime Python API offers additional features and helpers to enhance user experience. The main goal of Python API is to provide user-friendly and simple yet powerful tool for Python users.

## Easier Model Compilation

`CompiledModel` can be easily created with the helper method. It hides the creation of `Core` and applies `AUTO` inference mode by default.

```python
compiled_model = ov.compile_model(model)
```

## Model/CompiledModel Inputs and Outputs

Besides functions aligned to C++ API, some of them have their Python counterparts or extensions. For example, `Model` and `CompiledModel` inputs/outputs can be accessed via properties.

```python
import openvino.opset12 as ops

core = ov.Core()

input_a = ops.parameter([8], name="input_a")
res = ops.absolute(input_a)
model = ov.Model(res, [input_a])
compiled = core.compile_model(model, "CPU")
model.outputs[0].tensor.set_names({"result_0"})  # Add name for Output

print(model.inputs)
print(model.outputs)

print(compiled.inputs)
print(compiled.outputs)
```

Refer to <a href="../../../api/ie_python_api/api.html">Python API documentation</a>,
where helper functions or properties are available for different classes.

## Working with Tensor

Python API allows passing data as tensors. The `Tensor` object holds a copy of the data from the given array. The `dtype` of *numpy* arrays is converted to OpenVINO™ types automatically.

```python
data_float64 = np.ones(shape=(2,8))

tensor = ov.Tensor(data_float64)
assert tensor.element_type == ov.Type.f64

data_int32 = np.ones(shape=(2,8), dtype=np.int32)

tensor = ov.Tensor(data_int32)
assert tensor.element_type == ov.Type.i32
```

### Shared Memory Mode

`Tensor` objects can share the memory with *numpy* arrays. By specifying the `shared_memory` argument, the `Tensor` object does not copy data. Instead, it has access to the memory of the *numpy* array.

```python
data_to_share = np.ones(shape=(2,8))

shared_tensor = ov.Tensor(data_to_share, shared_memory=True)

# Editing of the numpy array affects Tensor's data
data_to_share[0][2] = 6.0
assert shared_tensor.data[0][2] == 6.0

# Editing of Tensor's data affects the numpy array
shared_tensor.data[0][2] = 0.6
assert data_to_share[0][2] == 0.6
```

## Running Inference

Python API supports extra calling methods to synchronous and asynchronous modes for inference.

All infer methods allow users to pass data as popular *numpy* arrays, gathered in either Python dicts or lists.

```python
# Passing inputs data in form of a dictionary
infer_request.infer(inputs={0: data})
# Passing inputs data in form of a list
infer_request.infer(inputs=[data])
```

Results from inference can be obtained in various ways:

```python
# Get output tensor
results = infer_request.get_output_tensor().data

# Get tensor with CompiledModel's output node
results = infer_request.get_tensor(compiled.outputs[0]).data

# Get all results with special helper property
results = list(infer_request.results.values())
```

### Synchronous Mode - Extended

Python API provides different synchronous calls to infer model, which block the application execution. Additionally, these calls return results of inference:

```python
# Simple call to InferRequest
results = infer_request.infer(inputs={0: data})
# Extra feature: calling CompiledModel directly
results = compiled_model(inputs={0: data})
```

### Inference Results - OVDict

Synchronous calls return a special data structure called `OVDict`. It can be compared to a "frozen dictionary". There are various ways of accessing the object's elements:

```python
results = compiled_model(inputs={0: data})

# Access via string
_ = results["result_0"]
# Access via index
_ = results[0]
# Access via output port
_ = results[compiled_model.outputs[0]]
# Use iterator over keys
_ = results[next(iter(results))]
# Iterate over values
_ = next(iter(results.values()))
```

:::note
It is possible to convert `OVDict` to a native dictionary using the `to_dict()` method.
:::

:::warning
Using `to_dict()` results in losing access via strings and integers. Additionally,
it performs a shallow copy, thus any modifications may affect the original
object as well.
:::

### AsyncInferQueue

Asynchronous mode pipelines can be supported with a wrapper class called `AsyncInferQueue`. This class automatically spawns the pool of `InferRequest` objects (also called "jobs") and provides synchronization mechanisms to control the flow of the pipeline.

Each job is distinguishable by a unique `id`, which is in the range from 0 up to the number of jobs specified in the `AsyncInferQueue` constructor.

The `start_async` function call is not required to be synchronized - it waits for any available job if the queue is busy/overloaded. Every `AsyncInferQueue` code block should end with the `wait_all` function which provides the "global" synchronization of all jobs in the pool and ensure that access to them is safe.

```python
core = ov.Core()

# Simple model that adds two inputs together
input_a = ops.parameter([8])
input_b = ops.parameter([8])
res = ops.add(input_a, input_b)
model = ov.Model(res, [input_a, input_b])
compiled = core.compile_model(model, "CPU")

# Number of InferRequests that AsyncInferQueue holds
jobs = 4
infer_queue = ov.AsyncInferQueue(compiled, jobs)

# Create data
data = [np.array([i] * 8, dtype=np.float32) for i in range(jobs)]

# Run all jobs
for i in range(len(data)):
    infer_queue.start_async({0: data[i], 1: data[i]})
infer_queue.wait_all()
```

:::warning
`InferRequest` objects that can be acquired by iterating over a `AsyncInferQueue` object or by `[id]` guaranteed to work with read-only methods like getting tensors.
Any mutating methods (e.g. start\_async, set\_callback) of a single request will put the parent AsyncInferQueue object in an invalid state.
:::

#### Acquiring Results from Requests

After the call to `wait_all`, jobs and their data can be safely accessed. Acquiring a specific job with `[id]` will return the `InferRequest` object, which will result in seamless retrieval of the output data.

```python
results = infer_queue[3].get_output_tensor().data
```

#### Setting Callbacks

Another feature of `AsyncInferQueue` is the ability to set callbacks. When callback is set, any job that ends inference calls upon the Python function. The callback function must have two arguments: one is the request that calls the callback, which provides the `InferRequest` API; the other is called "userdata", which provides the possibility of passing runtime values. Those values can be of any Python type and later used within the callback function.

The callback of `AsyncInferQueue` is uniform for every job. When executed, GIL is acquired to ensure safety of data manipulation inside the function.

```python
data_done = [False for _ in range(jobs)]

def f(request, userdata):
    print(f"Done! Result: {request.get_output_tensor().data}")
    data_done[userdata] = True

infer_queue.set_callback(f)

for i in range(len(data)):
    infer_queue.start_async({0: data[i], 1: data[i]}, userdata=i)
infer_queue.wait_all()

assert all(data_done)
```

### Working with u1, u4 and i4 Element Types

Since OpenVINO™ supports low precision element types, there are a few ways to handle them in Python.
To create an input tensor with such element types, you may need to pack your data in the new *numpy* array, with which the byte size matches the original input size:

```python
from openvino.helpers import pack_data

packed_buffer = pack_data(unt8_data, ov.Type.u4)
# Create tensor with shape in element types
t = ov.Tensor(packed_buffer, [100], ov.Type.u4)
```

To extract low precision values from a tensor into the *numpy* array, you can use the following helper:

```python
from openvino.helpers import unpack_data

unpacked_data = unpack_data(t.data, t.element_type, t.shape)
assert np.array_equal(unpacked_data , unt8_data)
```

### Release of GIL

Some functions in Python API release the Global Lock Interpreter (GIL) while running work-intensive code. This can help you achieve more parallelism in your application, using Python threads. For more information about GIL, refer to the <a href="../../../api/ie_python_api/api.html">Python API documentation</a>.

```python
import openvino as ov
from threading import Thread

input_data = []

# Processing input data will be done in a separate thread
# while compilation of the model and creation of the infer request
# is going to be executed in the main thread.
def prepare_data(input, image):
    shape = list(input.shape)
    resized_img = np.resize(image, shape)
    input_data.append(resized_img)

core = ov.Core()
model = core.read_model(model_path)
# Create thread with prepare_data function as target and start it
thread = Thread(target=prepare_data, args=[model.input(), image])
thread.start()
# The GIL will be released in compile_model.
# It allows a thread above to start the job,
# while main thread is running in the background.
compiled = core.compile_model(model, "CPU")
# After returning from compile_model, the main thread acquires the GIL
# and starts create_infer_request which releases it once again.
request = compiled.create_infer_request()
# Join the thread to make sure the input_data is ready
thread.join()
# running the inference
request.infer(input_data)
```

:::note
While GIL is released, functions can still modify and/or operate on Python objects in C++. Hence, there is no reference counting. You should pay attention to thread safety in case sharing of these objects with another thread occurs. It might affect code only if multiple threads are spawned in Python.
:::

#### List of Functions that Release the GIL

- openvino.AsyncInferQueue.start\_async
- openvino.AsyncInferQueue.is\_ready
- openvino.AsyncInferQueue.wait\_all
- openvino.AsyncInferQueue.get\_idle\_request\_id
- openvino.CompiledModel.create\_infer\_request
- openvino.CompiledModel.infer\_new\_request
- openvino.CompiledModel.\_\_call\_\_
- openvino.CompiledModel.export
- openvino.CompiledModel.get\_runtime\_model
- openvino.Core.compile\_model
- openvino.Core.read\_model
- openvino.Core.import\_model
- openvino.Core.query\_model
- openvino.Core.get\_available\_devices
- openvino.InferRequest.infer
- openvino.InferRequest.start\_async
- openvino.InferRequest.wait
- openvino.InferRequest.wait\_for
- openvino.InferRequest.get\_profiling\_info
- openvino.InferRequest.query\_state
- openvino.Model.reshape
- openvino.preprocess.PrePostProcessor.build
