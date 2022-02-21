# Heterogeneous execution {#openvino_docs_OV_UG_Hetero_execution}

## Introducing the Heterogeneous execution

The heterogeneous execution enables computing the inference of one model on several devices. The purposes of executing models in heterogeneous mode are to:

* Utilize the power of accelerators to process the heaviest parts of the model and to execute unsupported operations on fallback devices like the CPU
* Utilize all available hardware more efficiently during one inference

The execution through heterogeneous mode can be divided into two independent steps:

1. Setting of hardware affinity to operations (ov::Core::query_model is used internally by the Hetero device)
2. Compiling a model to the Heterogeneous device assuming splitting the model to parts and compiling on the specified devices (via ov::device::priorities), and executing them through the Heterogeneous plugin. The model is split to the subgraphs in according to the affinities where a set of conntected operations with the same affinity are supposed to be a dedicated subgraph.

These steps are decoupled. The setting of affinities can be done automatically using the `automatic fallback` policy or in `manual` mode:

- The fallback automatic policy causes "greedy" behavior and assigns all operations that can be executed on certain device according to the priorities you specify (for example, `ov::device::priorities("GPU,CPU")`).
Automatic policy does not take into account plugin peculiarities such as the inability to infer some layers without other special layers placed before or after that layer. The plugin is responsible for solving such cases. If the device plugin does not support the subgraph topology constructed by the HETERO plugin, then you should set affinity manually.
- Manual policy assumes explicit setting of affinities for all operations in the model using the runtime information ov::Node::get_rt_info.

### Details of Splitting Model and Execution

During compiling of the model in the Heterogeneous execution, the model is divided into separate parts and compiled on dedicated devices.
Intermediate tensors between these subgraphs are allocated automatically in the most efficient way.

### Sample Usage

OpenVINO™ sample programs can use the Heterogeneous execution used with the `-d` option:

```sh
./hello_classification <path_to_model>/squeezenet1.1.xml <path_to_pictures>/picture.jpg HETERO:GPU,CPU
```
where:
- `HETERO` stands for the Heterogeneous execution
- `GPU,CPU` points to fallback policy with priority on GPU and fallback to CPU

You can point more than two devices: `-d HETERO:MYRIAD,GPU,CPU`

### Defining and Configuring the Hetero Device

Following the OpenVINO™ convention of labeling devices, the Hetero execution uses the name `"HETERO"`. Configuration options for the Hetero device:

| Parameter name | C++ property | Parameter values | Default | Description |
| -------------- | ---------------- | ---------------- | --- | --- |
| "MULTI_DEVICE_PRIORITIES" | `ov::device::priorities` | comma-separated device names with no spaces | N/A | Prioritized list of devices |

### Annotation of Operations per Device and Default Fallback Policy

`Automatic fallback` policy decides which operation goes to which device automatically according to the support in dedicated devices (`GPU`, `CPU`, `MYRIAD`, etc) and query model step is called implicitly by Hetero device during model compilation:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_hetero.cpp
       :language: cpp
       :fragment: [compile_model]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_hetero.py
       :language: python
       :fragment: [compile_model]

@endsphinxdirective

Another way to annotate a model is to set all affinities `manually` using ov::Node::get_rt_info with key `"affinity"`:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_hetero.cpp
       :language: cpp
       :fragment: [set_manual_affinities]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_hetero.py
       :language: python
       :fragment: [set_manual_affinities]

@endsphinxdirective

The fallback policy does not work if at least one operation has an initialized `affinity`. If you want to adjust automatically set affinities, then get automatic affinities first, then fix them (usually, to minimize a number of total subgraphs to optimize memory transfers):

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_hetero.cpp
       :language: cpp
       :fragment: [fix_automatic_affinities]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_hetero.py
       :language: python
       :fragment: [fix_automatic_affinities]

@endsphinxdirective

> **NOTE**: ov::Core::query_model does not depend on affinities set by a user. Instead, it queries for an operation support based on device capabilities.

### Handling Difficult Topologies

Some topologies are not friendly to heterogeneous execution on some devices or cannot be executed at all with this device.
For example, models having activation operations that are not supported on the primary device are split by Hetero device into multiple set of subgraphs which leads to unoptimal execution.
If transmitting data from one subgraph of a whole model to another part in heterogeneous mode takes more time than in normal execution, it may not make sense to execute them heterogeneously.
In this case, you can define the heaviest part manually and set the affinity to avoid sending data back and forth many times during one inference.

### Configure fallback devices
If you want different devices in Hetero execution to have different device-specific configuration options, you can use the special helper property ov::device::properties:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_hetero.cpp
       :language: cpp
       :fragment: [configure_fallback_devices]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_hetero.py
       :language: python
       :fragment: [configure_fallback_devices]

@endsphinxdirective

In the example above, `CPU` device is configured to enable profiling data, while only `GPU` device has configuration property to perform inference in `f16` precision, while CPU has default execution precision.

### Analyzing Performance Heterogeneous Execution
After enabling the <code>OPENVINO_HETERO_VISUALIZE</code> environment variable, you can dump GraphViz* `.dot` files with annotations of operations per devices.

The Heterogeneous device can generate two files:

* `hetero_affinity_<model name>.dot` - annotation of affinities per operation.
* `hetero_subgraphs_<model name>.dot` - annotation of affinities per graph.

You can use the GraphViz* utility or a file converter to view the images. On the Ubuntu* operating system, you can use xdot:

* `sudo apt-get install xdot`
* `xdot hetero_subgraphs.dot`

You can use performance data (in sample applications, it is the option `-pc`) to get the performance data on each subgraph.

Here is an example of the output for Googlenet v1 running on HDDL with fallback to CPU:

```
subgraph1: 1. input preprocessing (mean data/HDDL):EXECUTED layerType:          realTime: 129   cpu: 129  execType:
subgraph1: 2. input transfer to DDR:EXECUTED                layerType:          realTime: 201   cpu: 0    execType:
subgraph1: 3. HDDL execute time:EXECUTED                    layerType:          realTime: 3808  cpu: 0    execType:
subgraph1: 4. output transfer from DDR:EXECUTED             layerType:          realTime: 55    cpu: 0    execType:
subgraph1: 5. HDDL output postprocessing:EXECUTED           layerType:          realTime: 7     cpu: 7    execType:
subgraph1: 6. copy to IE blob:EXECUTED                      layerType:          realTime: 2     cpu: 2    execType:
subgraph2: out_prob:          NOT_RUN                       layerType: Output   realTime: 0     cpu: 0    execType: unknown
subgraph2: prob:              EXECUTED                      layerType: SoftMax  realTime: 10    cpu: 10   execType: ref
Total time: 4212 microseconds
```
### See Also
[Supported Devices](Supported_Devices.md)
