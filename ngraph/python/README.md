## nGraph Compiler stack

[nGraph][ngraph_github] is an open-source graph compiler for Artificial 
Neural Networks (ANNs). The nGraph Compiler stack provides an inherently 
efficient graph-based compilation infrastructure designed to be compatible 
with the many upcoming processors, like the Intel Nervana&trade; Neural Network 
Processor (Intel&reg; Nervana&trade; NNP), while also unlocking a massive performance 
boost on any existing hardware targets in your neural network: both GPUs 
and CPUs. Using its flexible infrastructure, you will find it becomes 
much easier to create Deep Learning (DL) models that can adhere to the 
"write once, run anywhere" mantra that enables your AI solutions to easily
go from concept to production to scale.

Frameworks using nGraph to execute workloads have shown [up to 45X] performance 
boost compared to native implementations.

### Using the Python API 

nGraph can be used directly with the [Python API][api_python] described here, or 
with the [C++ API][api_cpp] described in the [core documentation]. Alternatively, 
its performance benefits can be realized through frontends such as 
[TensorFlow][frontend_tf], [PaddlePaddle][paddle_paddle] and [ONNX][frontend_onnx].
You can also create your own custom framework to integrate directly with the 
[nGraph Ops] for highly-targeted graph execution.

## Installation

nGraph is available as binary wheels you can install from PyPI. nGraph binary 
wheels are currently tested on Ubuntu 16.04. To build and test on other 
systems, you may want to try [building][ngraph_building] from sources.

Installing nGraph Python API from PyPI is easy:

    pip install ngraph-core

## Usage example

Using nGraph's Python API to construct a computation graph and execute a 
computation is simple. The following example shows how to create a minimal 
`(A + B) * C` computation graph and calculate a result using 3 numpy arrays 
as input.


```python
import numpy as np
import ngraph as ng

A = ng.parameter(shape=[2, 2], name='A', dtype=np.float32)
B = ng.parameter(shape=[2, 2], name='B', dtype=np.float32)
C = ng.parameter(shape=[2, 2], name='C', dtype=np.float32)
# >>> print(A)
# <Parameter: 'A' ([2, 2], float)>

model = (A + B) * C
# >>> print(model)
# <Multiply: 'Multiply_14' ([2, 2])>

runtime = ng.runtime(backend_name='CPU')
# >>> print(runtime)
# <Runtime: Backend='CPU'>

computation = runtime.computation(model, A, B, C)
# >>> print(computation)
# <Computation: Multiply_14(A, B, C)>

value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
value_c = np.array([[9, 10], [11, 12]], dtype=np.float32)

result = computation(value_a, value_b, value_c)
# >>> print(result)
# [[ 54.  80.]
#  [110. 144.]]

print('Result = ', result)
```

[up to 45X]: https://ai.intel.com/ngraph-compiler-stack-beta-release/
[frontend_onnx]: https://pypi.org/project/ngraph-onnx/
[paddle_paddle]: https://ngraph.nervanasys.com/docs/latest/frameworks/paddle_integ.html 
[frontend_tf]: https://pypi.org/project/ngraph-tensorflow-bridge/
[ngraph_github]: https://github.com/NervanaSystems/ngraph "nGraph on GitHub"
[ngraph_building]: https://github.com/NervanaSystems/ngraph/blob/master/python/BUILDING.md "Building nGraph"
[api_python]: https://ngraph.nervanasys.com/docs/latest/python_api/ "nGraph's Python API documentation"
[api_cpp]: https://ngraph.nervanasys.com/docs/latest/backend-support/cpp-api.html
[core documentation]: https://ngraph.nervanasys.com/docs/latest/core/overview.html
[nGraph Ops]: http://ngraph.nervanasys.com/docs/latest/ops/index.html


