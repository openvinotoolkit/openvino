---
sidebar_position: 3
sidebar_label: "OpenVINO™ Runtime Python API Advanced Inference"
description: "OpenVINO™ Runtime Python API enables you to share memory on inputs, hide the latency with asynchronous calls and implement \"postponed return\"."
---

# OpenVINO™ Runtime Python API Advanced Inference

:::warning
All methods described here are highly dependent on the specific hardware and software setup.
Consider conducting your own experiments with various models and different input/output
sizes. The methods presented here are not universal, they may or may not apply to the
specific pipeline. Consider all trade-offs and avoid premature optimizations.
:::

## Direct Inference with `CompiledModel`

The `CompiledModel` class provides the `__call__` method that runs a single synchronous
inference using the given model. In addition to a compact code, all future calls
to `CompiledModel.__call__` will result in less overhead,
as the object reuses the already created `InferRequest`.

```python
# Calling CompiledModel creates and saves InferRequest object
results_0 = compiled_model({"input_0": data_0, "input_1": data_1})
# Second call reuses previously created InferRequest object
results_1 = compiled_model({"input_0": data_2, "input_1": data_3})
```

## Shared Memory on Inputs and Outputs

While using `CompiledModel`, `InferRequest` and `AsyncInferQueue`,
the OpenVINO™ Runtime Python API provides an additional mode: **Shared Memory**.
Specify the `share_inputs` and `share_outputs` flag to enable or disable this feature.
The "Shared Memory" mode may be beneficial when inputs or outputs are large and copying data is considered an expensive operation.

This feature creates shared `Tensor`
instances with the zero-copy approach, reducing overhead of setting inputs
to minimum. For outputs this feature creates numpy views on data. Example usage:

```python
# Data can be shared only on inputs
_ = compiled_model({"input_0": data_0, "input_1": data_1}, share_inputs=True)
_ = request.infer({"input_0": data_0, "input_1": data_1}, share_inputs=True)
# Data can be shared only on outputs
_ = request.infer({"input_0": data_0, "input_1": data_1}, share_outputs=True)
# Or both flags can be combined to achieve desired behavior
_ = compiled_model({"input_0": data_0, "input_1": data_1}, share_inputs=False, share_outputs=True)
```

Shared Memory on inputs is enabled by default in `CompiledModel.__call__`.
For other methods, like `InferRequest.infer` or `InferRequest.start_async`,
it is required to set the flag to `True` manually.

Shared Memory on outputs is disabled by default in all sequential inference methods
(`CompiledModel.__call__` and `InferRequest.infer`). It is required to set the flag to `True` manually.

:::warning
When data is being shared, all modifications, including subsequent inference calls,
may affect inputs and outputs of the inference.
Use this feature with caution, especially in multi-threaded/parallel code,
where data can be modified outside of the function's control flow.
:::

## Hiding Latency with Asynchronous Calls

Asynchronous calls allow to hide latency to optimize overall runtime of a codebase.
For example, `InferRequest.start_async` releases the GIL and provides non-blocking call.
It is beneficial to process other calls while waiting to finish compute-intensive inference.
Example usage:

```python
import time

# Long running function
def run(time_in_sec):
    time.sleep(time_in_sec)

# No latency hiding
results = request.infer({"input_0": data_0, "input_1": data_1})[0]
run(time_in_sec)

# Hiding latency
request.start_async({"input_0": data_0, "input_1": data_1})
run(time_in_sec)
request.wait()
results = request.get_output_tensor(0).data  # Gather data from InferRequest
```

:::note
It is up to the user/developer to optimize the flow in a codebase to benefit from potential parallelization.
:::

## Postponed Return with Asynchronous Calls

Postponed Return is a practice to reduce overhead of returning `OVDict` from
synchronous calls. Postponed Return is useful in the following cases:

- only a part of output data is required. For example, only one specific output is significant in a given pipeline step and all outputs are large and expensive to copy.
- data is not required immediately. For example, it can be later extracted inside the pipeline as a part of latency hiding.
- data return is not required at all. For example, models are being chained with the pure `Tensor` interface.

```python
# Standard approach
results = request.infer({"input_0": data_0, "input_1": data_1})[0]

# "Postponed Return" approach
request.start_async({"input_0": data_0, "input_1": data_1})
request.wait()
results = request.get_output_tensor(0).data  # Gather data "on demand" from InferRequest
```
