# Heterogeneous execution {#openvino_docs_OV_UG_Hetero_execution}

Heterogeneous execution enables executing inference of one model on several devices. Its purpose is to:

* Utilize the power of accelerators to process the heaviest parts of the model and to execute unsupported operations on fallback devices, like the CPU.
* Utilize all available hardware more efficiently during one inference.

Execution via the heterogeneous mode can be divided into two independent steps:

1. Setting hardware affinity to operations (`ov::Core::query_model` is used internally by the Hetero device)
2. Compiling a model to the Heterogeneous device assumes splitting the model to parts, compiling them on the specified devices (via `ov::device::priorities`), and executing them in the Heterogeneous mode. The model is split to subgraphs in accordance with the affinities, where a set of connected operations with the same affinity is to be a dedicated subgraph. Each subgraph is compiled on a dedicated device and multiple `ov::CompiledModel` objects are made, which are connected via automatically allocated intermediate tensors.

These two steps are not interconnected and affinities can be set in one of two ways, used separately or in combination (as described below): in the `manual` or the `automatic` mode.

### Defining and Configuring the Hetero Device

Following the OpenVINO™ naming convention, the Hetero execution plugin is assigned the label of `"HETERO".` It may be defined with no additional parameters, resulting in defaults being used, or configured further with the following setup options: 

@sphinxdirective
+-------------------------------+--------------------------------------------+-----------------------------------------------------------+
| Parameter Name & C++ property | Property values                            | Description                                               |
+===============================+============================================+===========================================================+
| | "MULTI_DEVICE_PRIORITIES"   | | HETERO: <device names>                   | | Lists the devices available for selection.              |
| | `ov::device::priorities`    | | comma-separated, no spaces               | | The device sequence will be taken as priority           |
| |                             | |                                          | | from high to low.                                       |
+-------------------------------+--------------------------------------------+-----------------------------------------------------------+
@endsphinxdirective

### Manual and Automatic modes for assigning affinities

#### The Manual Mode
It assumes setting affinities explicitly for all operations in the model using `ov::Node::get_rt_info` with the `"affinity"` key. 

If you assign specific operation to a specific device, make sure that the device actually supports the operation. 
Randomly selecting operations and setting affinities may lead to decrease in model accuracy. To avoid that, try to set the related operations or subgraphs of this operation to the same affinity, such as the constant operation that will be folded into this operation.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_hetero.cpp set_manual_affinities

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_hetero.py set_manual_affinities

@endsphinxtab

@endsphinxtabset




#### The Automatic Mode
It decides automatically which operation is assigned to which device according to the support from dedicated devices (`GPU`, `CPU`, `GNA`, etc.) and query model step is called implicitly by Hetero device during model compilation.

The automatic mode causes "greedy" behavior and assigns all operations that can be executed on a given device to it, according to the priorities you specify (for example, `ov::device::priorities("GPU,CPU")`).
It does not take into account device peculiarities such as the inability to infer certain operations without other special operations placed before or after that layer. If the device plugin does not support the subgraph topology constructed by the HETERO device, then you should set affinity manually.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_hetero.cpp compile_model

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_hetero.py compile_model

@endsphinxtab

@endsphinxtabset

#### Using Manual and Automatic Modes in Combination
In some cases you may need to consider manually adjusting affinities which were set automatically. It usually serves minimizing the number of total subgraphs to optimize memory transfers. To do it, you need to "fix" the automatically assigned affinities like so:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_hetero.cpp fix_automatic_affinities

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_hetero.py fix_automatic_affinities

@endsphinxtab

@endsphinxtabset

Importantly, the automatic mode will not work if any operation in a model has its `"affinity"` already initialized.

> **NOTE**: `ov::Core::query_model` does not depend on affinities set by a user. Instead, it queries for an operation support based on device capabilities.

### Configure fallback devices
If you want different devices in Hetero execution to have different device-specific configuration options, you can use the special helper property `ov::device::properties`:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_hetero.cpp configure_fallback_devices

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_hetero.py configure_fallback_devices

@endsphinxtab

@endsphinxtabset

In the example above, the `GPU` device is configured to enable profiling data and uses the default execution precision, while `CPU` has the configuration property to perform inference in `fp32`.

### Handling of Difficult Topologies

Some topologies are not friendly to heterogeneous execution on some devices, even to the point of being unable to execute.
For example, models having activation operations that are not supported on the primary device are split by Hetero into multiple sets of subgraphs which leads to suboptimal execution.
If transmitting data from one subgraph to another part of the model in the heterogeneous mode takes more time than under normal execution, heterogeneous execution may be unsubstantiated.
In such cases, you can define the heaviest part manually and set the affinity to avoid sending data back and forth many times during one inference.

### Analyzing Performance of Heterogeneous Execution
After enabling the <code>OPENVINO_HETERO_VISUALIZE</code> environment variable, you can dump GraphViz `.dot` files with annotations of operations per devices.

The Heterogeneous execution mode can generate two files:

* `hetero_affinity_<model name>.dot` - annotation of affinities per operation.
* `hetero_subgraphs_<model name>.dot` - annotation of affinities per graph.

You can use the GraphViz utility or a file converter to view the images. On the Ubuntu operating system, you can use xdot:

* `sudo apt-get install xdot`
* `xdot hetero_subgraphs.dot`

You can use performance data (in sample applications, it is the option `-pc`) to get the performance data on each subgraph.

Here is an example of the output for Googlenet v1 running on HDDL (device no longer supported) with fallback to CPU:

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
### Sample Usage

OpenVINO™ sample programs can use the Heterogeneous execution used with the `-d` option:

```sh
./hello_classification <path_to_model>/squeezenet1.1.xml <path_to_pictures>/picture.jpg HETERO:GPU,CPU
```
where:
- `HETERO` stands for the Heterogeneous execution
- `GPU,CPU` points to a fallback policy with the priority on GPU and fallback to CPU

You can also point to more than two devices: `-d HETERO:GNA,GPU,CPU`

### Additional Resources

* [Supported Devices](supported_plugins/Supported_Devices.md)
