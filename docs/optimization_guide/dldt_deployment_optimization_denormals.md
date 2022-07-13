# Denormals Optimization {#openvino_docs_deployment_optimization_guide_denormals}
## Denormal Number
Denormal number is non-zero, finite float number that is very close to zero, i.e. the numbers in (0, 1.17549e-38] and (0, -1.17549e-38). In such case, normalized-number encodeing format does not have capability to encode the number and underflow will happen. The computation involving this kind of numbers is extremly slow on many hardware.

## Optimize Denormals
As denormal number is extremly close to zero, treating denormal as zero directly is a straightforward and simple method to optimize denormals computation. As this optimization does not comply with IEEE standard 754, in case it introduce unacceptable accuracy degradation, a propery(ov::denormals_optimization) is introduced to control this behavior. If there are denormal numbers in users' use case, and see no or ignorable accuracy drop, we could set this property to "YES" to improve performance, otherwise set this to "NO". If it's not set explicitly by property, this optimization is disabled by default if application program also does not perform denormals optimization. After this property is turned on, OpenVINO will provide an cross operation-system/compiler and safe optimization on all platform when applicable.
There are cases that application program where OpenVINO is used also perform this low-level denormals optimization. If it's optimized by setting FLZ(Flush-To-Zero) and DAZ(Denormals-As-Zero) flag in MXCSR register in the begining of thread where OpenVINO is called, OpenVINO will inherite this setting in the same thread and sub-thread, and then no need set with property. In this case, application program users should be responsible for the effectiveness and safty of the settings.
It need also to be mentioned that this property should must be set before 'compile_model()', and for now only take effect on CPU.

## example
To enable denormals optimization, the application must set "YES" to the ov::denormals_optimization property:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_denormals.cpp
         :language: cpp
         :fragment: [ov:denormals_optimization:part0]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_denormals.py
         :language: python
         :fragment: [ov:denormals_optimization:part0]

@endsphinxdirective
