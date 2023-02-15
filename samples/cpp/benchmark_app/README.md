# Benchmark C++ Tool {#openvino_inference_engine_samples_benchmark_app_README}

This page demonstrates how to use the Benchmark C++ Tool to estimate deep learning inference performance on supported devices.

> **NOTE**: This page describes usage of the C++ implementation of the Benchmark Tool. For the Python implementation, refer to the [Benchmark Python Tool](../../../tools/benchmark_tool/README.md) page. The Python version is recommended for benchmarking models that will be used in Python applications, and the C++ version is recommended for benchmarking models that will be used in C++ applications. Both tools have a similar command interface and backend.


## Basic Usage
To use the C++ benchmark_app, you must first build it following the [Build the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) instructions and then set up paths and environment variables by following the [Get Ready for Running the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) instructions. Navigate to the directory where the benchmark_app C++ sample binary was built.

> **NOTE**: If you installed OpenVINO Runtime using PyPI or Anaconda Cloud, only the [Benchmark Python Tool](../../../tools/benchmark_tool/README.md) is available, and you should follow the usage instructions on that page instead.

The benchmarking application works with models in the OpenVINO IR (`model.xml` and `model.bin`) and ONNX (`model.onnx`) formats. Make sure to [convert your models](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) if necessary.

To run benchmarking with default options on a model, use the following command:

```
./benchmark_app -m model.xml
```

By default, the application will load the specified model onto the CPU and perform inferencing on batches of randomly-generated data inputs for 60 seconds. As it loads, it prints information about benchmark parameters. When benchmarking is completed, it reports the minimum, average, and maximum inferencing latency and average the throughput.

You may be able to improve benchmark results beyond the default configuration by configuring some of the execution parameters for your model. For example, you can use "throughput" or "latency" performance hints to optimize the runtime for higher FPS or reduced inferencing time. Read on to learn more about the configuration options available with benchmark_app.

## Configuration Options
The benchmark app provides various options for configuring execution parameters. This section covers key configuration options for easily tuning benchmarking to achieve better performance on your device. A list of all configuration options is given in the [Advanced Usage](#advanced-usage-cpp-benchmark) section.

### Performance hints: latency and throughput
The benchmark app allows users to provide high-level "performance hints" for setting latency-focused or throughput-focused inference modes. This hint causes the runtime to automatically adjust runtime parameters, such as the number of processing streams and inference batch size, to prioritize for reduced latency or high throughput.

The performance hints do not require any device-specific settings and they are completely portable between devices. Parameters are automatically configured based on whichever device is being used. This allows users to easily port applications between hardware targets without having to re-determine the best runtime parameters for the new device.

If not specified, throughput is used as the default. To set the hint explicitly, use `-hint latency` or `-hint throughput` when running benchmark_app:

```
./benchmark_app -m model.xml -hint latency
./benchmark_app -m model.xml -hint throughput
```

> **NOTE**
It is up to the user to ensure the environment on which the benchmark is running is optimized for maximum performance.
Otherwise, different results may occur when using the application in different environment settings (such as power optimization settings, processor overclocking, thermal throttling).

#### Latency
Latency is the amount of time it takes to process a single inference request. In applications where data needs to be inferenced and acted on as quickly as possible (such as autonomous driving), low latency is desirable. For conventional devices, lower latency is achieved by reducing the amount of parallel processing streams so the system can utilize as many resources as possible to quickly calculate each inference request. However, advanced devices like multi-socket CPUs and modern GPUs are capable of running multiple inference requests while delivering the same latency.

When benchmark_app is run with `-hint latency`, it determines the optimal number of parallel inference requests for minimizing latency while still maximizing the parallelization capabilities of the hardware. It automatically sets the number of processing streams and inference batch size to achieve the best latency.

#### Throughput
Throughput is the amount of data an inferencing pipeline can process at once, and it is usually measured in frames per second (FPS) or inferences per second. In applications where large amounts of data needs to be inferenced simultaneously (such as multi-camera video streams), high throughput is needed. To achieve high throughput, the runtime focuses on fully saturating the device with enough data to process. It utilizes as much memory and as many parallel streams as possible to maximize the amount of data that can be processed simultaneously.

When benchmark_app is run with `-hint throughput`, it maximizes the number of parallel inference requests to utilize all the threads available on the device. On GPU, it automatically sets the inference batch size to fill up the GPU memory available.

For more information on performance hints, see the [High-level Performance Hints](../../../docs/OV_Runtime_UG/performance_hints.md) page. For more details on optimal runtime configurations and how they are automatically determined using performance hints, see [Runtime Inference Optimizations](../../../docs/optimization_guide/dldt_deployment_optimization_guide.md).


### Device
To set which device benchmarking runs on, use the `-d <device>` argument. This will tell benchmark_app to run benchmarking on that specific device. The benchmark app supports "CPU", "GPU", and "GNA" devices. In order to use the GPU or GNA, the system must have the appropriate drivers installed. If no device is specified, benchmark_app will default to using CPU.

For example, to run benchmarking on GPU, use:

```
./benchmark_app -m model.xml -d GPU
```

You may also specify "AUTO" as the device, in which case the benchmark_app will automatically select the best device for benchmarking and support it with the CPU at the model loading stage. This may result in increased performance, thus, should be used purposefully. For more information, see the [Automatic device selection](../../../docs/OV_Runtime_UG/auto_device_selection.md) page.

(Note: If the latency or throughput hint is set, it will automatically configure streams and batch sizes for optimal performance based on the specified device.)

### Number of iterations
By default, the benchmarking app will run for a predefined duration, repeatedly performing inferencing with the model and measuring the resulting inference speed. There are several options for setting the number of inference iterations:

* Explicitly specify the number of iterations the model runs using the `-niter <number_of_iterations>` option.
* Set how much time the app runs for using the `-t <seconds>` option.
* Set both of them (execution will continue until both conditions are met).
* If neither -niter nor -t are specified, the app will run for a predefined duration that depends on the device.

The more iterations a model runs, the better the statistics will be for determing average latency and throughput.

### Inputs
The benchmark tool runs benchmarking on user-provided input images in `.jpg`, `.bmp`, or `.png` format. Use `-i <PATH_TO_INPUT>` to specify the path to an image, or folder of images. For example, to run benchmarking on an image named `test1.jpg`, use:

```
./benchmark_app -m model.xml -i test1.jpg
```

The tool will repeatedly loop through the provided inputs and run inferencing on them for the specified amount of time or number of iterations. If the `-i` flag is not used, the tool will automatically generate random data to fit the input shape of the model. 

### Examples
For more usage examples (and step-by-step instructions on how to set up a model for benchmarking), see the [Examples of Running the Tool](#examples-of-running-the-tool-cpp) section.

## <a name="advanced-usage-cpp-benchmark"></a> Advanced Usage

> **NOTE**: By default, OpenVINO samples, tools and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channel order in the sample or demo application or reconvert your model using the Model Optimizer tool with --reverse_input_channels argument specified. For more information about the argument, refer to When to Reverse Input Channels section of Converting a Model to Intermediate Representation (IR).

### Per-layer performance and logging
The application also collects per-layer Performance Measurement (PM) counters for each executed infer request if you enable statistics dumping by setting the `-report_type` parameter to one of the possible values:

* `no_counters` report includes configuration options specified, resulting FPS and latency.
* `average_counters` report extends the `no_counters` report and additionally includes average PM counters values for each layer from the network.
* `detailed_counters` report extends the `average_counters` report and additionally includes per-layer PM counters and latency for each executed infer request.

Depending on the type, the report is stored to benchmark_no_counters_report.csv, benchmark_average_counters_report.csv, or benchmark_detailed_counters_report.csv file located in the path specified in -report_folder. The application also saves executable graph information serialized to an XML file if you specify a path to it with the -exec_graph_path parameter.

### <a name="all-configuration-options-cpp-benchmark"></a> All configuration options

Running the application with the `-h` or `--help` option yields the following usage message:

```
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
usage: benchmark_app [OPTION]

Options:
    -h, --help                    Print the usage message
    -m  <path>                    Required. Path to an .xml/.onnx file with a trained model or to a .blob files with a trained compiled model.
    -i  <path>                    Optional. Path to a folder with images and/or binaries or to specific image or binary file.
                              In case of dynamic shapes models with several inputs provide the same number of files for each input (except cases with single file for any input):"input1:1.jpg input2:1.bin", "input1:1.bin,2.bin input2:3.bin input3:4.bin,5.bin ". Also you can pass specific keys for inputs: "random" - for fillling input with random data, "image_info" - for filling input with image size.
                              You should specify either one files set to be used for all inputs (without providing input names) or separate files sets for every input of model (providing inputs names).
                              Currently supported data types: bmp, bin, npy.
                              If OPENCV is enabled, this functionality is extended with the following data types:
                              dib, jpeg, jpg, jpe, jp2, png, pbm, pgm, ppm, sr, ras, tiff, tif.
    -d  <device>                  Optional. Specify a target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. Use "-d MULTI:<comma-separated_devices_list>" format to specify MULTI plugin. The application looks for a suitable plugin for the specified device.
    -hint  <performance hint> (latency or throughput or cumulative_throughput or none)   Optional. Performance hint allows the OpenVINO device to select the right model-specific settings.
                               'throughput' or 'tput': device performance mode will be set to THROUGHPUT.
                               'cumulative_throughput' or 'ctput': device performance mode will be set to CUMULATIVE_THROUGHPUT.
                               'latency': device performance mode will be set to LATENCY.
                               'none': no device performance mode will be set.
                              Using explicit 'nstreams' or other device-specific options, please set hint to 'none'
    -niter  <integer>             Optional. Number of iterations. If not specified, the number of iterations is calculated depending on a device.
    -t                            Optional. Time in seconds to execute topology.

Input shapes
    -b  <integer>                 Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation.
    -shape                        Optional. Set shape for model input. For example, "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]" in case of one input size. This parameter affect model input shape and can be dynamic. For dynamic dimensions use symbol `?` or '-1'. Ex. [?,3,?,?]. For bounded dimensions specify range 'min..max'. Ex. [1..10,3,?,?].
    -data_shape                   Required for models with dynamic shapes. Set shape for input blobs. In case of one input size: "[1,3,224,224]" or "input1[1,3,224,224],input2[1,4]". In case of several input sizes provide the same number for each input (except cases with single shape for any input): "[1,3,128,128][3,3,128,128][1,3,320,320]", "input1[1,1,128,128][1,1,256,256],input2[80,1]" or "input1[1,192][1,384],input2[1,192][1,384],input3[1,192][1,384],input4[1,192][1,384]". If model shapes are all static specifying the option will cause an exception.
    -layout                       Optional. Prompts how model layouts should be treated by application. For example, "input1[NCHW],input2[NC]" or "[NCHW]" in case of one input size.

Advanced options
    -extensions  <absolute_path>  Required for custom layers (extensions). Absolute path to a shared library with the kernels implementations.
    -c  <absolute_path>           Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
    -cache_dir  <path>            Optional. Enables caching of loaded models to specified directory. List of devices which support caching is shown at the end of this message.
    -load_from_file               Optional. Loads model from file directly without read_model. All CNNNetwork options (like re-shape) will be ignored
    -api <sync/async>             Optional (deprecated). Enable Sync/Async API. Default value is "async".
    -nireq  <integer>             Optional. Number of infer requests. Default value is determined automatically for device.
    -nstreams  <integer>          Optional. Number of streams to use for inference on the CPU or GPU devices (for HETERO and MULTI device cases use format <dev1>:<nstreams1>,<dev2>:<nstreams2> or just <nstreams>). Default value is determined automatically for a device.Please note that although the automatic selection usually provides a reasonable performance, it still may be non - optimal for some cases, especially for very small models. See sample's README for more details. Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency estimations the number of streams should be set to 1.
    -inference_only         Optional. Measure only inference stage. Default option for static models. Dynamic models are measured in full mode which includes inputs setup stage, inference only mode available for them with single input data shape only. To enable full mode for static models pass "false" value to this argument: ex. "-inference_only=false".
    -infer_precision        Optional. Specifies the inference precision. Example #1: '-infer_precision bf16'. Example #2: '-infer_precision CPU:bf16,GPU:f32'

Preprocessing options:
    -ip   <value>           Optional. Specifies precision for all input layers of the model.
    -op   <value>           Optional. Specifies precision for all output layers of the model.
    -iop  <value>           Optional. Specifies precision for input and output layers by name.
                                             Example: -iop "input:FP16, output:FP16".
                                             Notice that quotes are required.
                                             Overwrites precision from ip and op options for specified layers.
    -mean_values   [R,G,B]  Optional. Mean values to be used for the input image per channel. Values to be provided in the [R,G,B] format. Can be defined for desired input of the model, for example: "--mean_values data[255,255,255],info[255,255,255]". The exact meaning and order of channels depend on how the original model was trained. Applying the values affects performance and may cause type conversion
    -scale_values  [R,G,B]  Optional. Scale values to be used for the input image per channel. Values are provided in the [R,G,B] format. Can be defined for desired input of the model, for example: "--scale_values data[255,255,255],info[255,255,255]". The exact meaning and order of channels depend on how the original model was trained. If both --mean_values and --scale_values are specified, the mean is subtracted first and then scale is applied regardless of the order of options in command line. Applying the values affects performance and may cause type conversion

Device-specific performance options:
    -nthreads  <integer>          Optional. Number of threads to use for inference on the CPU (including HETERO and MULTI cases).
    -pin  <string>  ("YES"|"CORE") / "HYBRID_AWARE" / ("NO"|"NONE") / "NUMA"  Optional. Explicit inference threads binding options (leave empty to let the OpenVINO make a choice):
                                enabling threads->cores pinning("YES", which is already default for any conventional CPU),
                                letting the runtime to decide on the threads->different core types("HYBRID_AWARE", which is default on the hybrid CPUs)
                                threads->(NUMA)nodes("NUMA") or
                                completely disable("NO") CPU inference threads pinning

Statistics dumping options:
    -latency_percentile     Optional. Defines the percentile to be reported in latency metric. The valid range is [1, 100]. The default value is 50 (median).
    -report_type  <type>    Optional. Enable collecting statistics report. "no_counters" report contains configuration options specified, resulting FPS and latency. "average_counters" report extends "no_counters" report and additionally includes average PM counters values for each layer from the model. "detailed_counters" report extends "average_counters" report and additionally includes per-layer PM counters and latency for each executed infer request.
    -report_folder          Optional. Path to a folder where statistics report is stored.
    -json_stats             Optional. Enables JSON-based statistics output (by default reporting system will use CSV format). Should be used together with -report_folder option.
    -pc                     Optional. Report performance counters.
    -pcsort                 Optional. Report performance counters and analysis the sort hotpoint opts.  "sort" Analysis opts time cost, print by hotpoint order  "no_sort" Analysis opts time cost, print by normal order  "simple_sort" Analysis opts time cost, only print EXECUTED opts by normal order
    -pcseq                  Optional. Report latencies for each shape in -data_shape sequence.
    -exec_graph_path        Optional. Path to a file where to store executable graph information serialized.
    -dump_config            Optional. Path to JSON file to dump IE parameters, which were set by application.
    -load_config            Optional. Path to JSON file to load custom IE parameters. Please note, command line parameters have higher priority then parameters from configuration file.
                              Example 1: a simple JSON file for HW device with primary properties.
                                       {
                                            "CPU": {"NUM_STREAMS": "3", "PERF_COUNT": "NO"}
                                       }
                              Example 2: a simple JSON file for meta device(AUTO/MULTI) with HW device properties.
                                       {
                                               "AUTO": {
                                                       "PERFORMANCE_HINT": "",
                                                       "PERF_COUNT": "NO",
                                                       "DEVICE_PROPERTIES": {
                                                       "CPU": {
                                                           "INFERENCE_PRECISION_HINT": "f32",
                                                           "NUM_STREAMS": "3"
                                                       },
                                                       "GPU": {
                                                           "INFERENCE_PRECISION_HINT": "f32",
                                                           "NUM_STREAMS": "5"
                                                       }
                                                   }
                                               }
                                       }
```

Running the application with the empty list of options yields the usage message given above and an error message.

### More information on inputs
The benchmark tool supports topologies with one or more inputs. If a topology is not data sensitive, you can skip the input parameter, and the inputs will be filled with random values. If a model has only image input(s), provide a folder with images or a path to an image as input. If a model has some specific input(s) (besides images), please prepare a binary file(s) or numpy array(s) that is filled with data of appropriate precision and provide a path to it as input. If a model has mixed input types, the input folder should contain all required files. Image inputs are filled with image files one by one. Binary inputs are filled with binary inputs one by one.

## <a name="examples-of-running-the-tool-cpp"></a> Examples of Running the Tool
This section provides step-by-step instructions on how to run the Benchmark Tool with the `asl-recognition` model from the [Open Model Zoo](@ref model_zoo) on CPU or GPU devices. It uses random data as the input.

> **NOTE**: Internet access is required to execute the following steps successfully. If you have access to the Internet through a proxy server only, please make sure that it is configured in your OS environment.

1. Install OpenVINO Development Tools (if it hasn't been installed already):
   ```sh
   pip install openvino-dev
   ```

2. Download the model using `omz_downloader`, specifying the model name and directory to download the model to:
   ```sh
   omz_downloader --name asl-recognition-0004 --precisions FP16 --output_dir omz_models
   ```

3. Run the tool, specifying the location of the model .xml file, the device to perform inference on, and with a performance hint. The following commands demonstrate examples of how to run the Benchmark Tool in latency mode on CPU and throughput mode on GPU devices:

   * On CPU (latency mode):
   ```sh
   ./benchmark_app -m omz_models/intel/asl-recognition-0004/FP16/asl-recognition-0004.xml -d CPU -hint latency
   ```

   * On GPU (throughput mode):
   ```sh
   ./benchmark_app -m omz_models/intel/asl-recognition-0004/FP16/asl-recognition-0004.xml -d GPU -hint throughput
   ```

The application outputs the number of executed iterations, total duration of execution, latency, and throughput.
Additionally, if you set the `-report_type` parameter, the application outputs a statistics report. If you set the `-pc` parameter, the application outputs performance counters. If you set `-exec_graph_path`, the application reports executable graph information serialized. All measurements including per-layer PM counters are reported in milliseconds.

An example of the information output when running benchmark_app on CPU in latency mode is shown below:

   ```sh
   ./benchmark_app -m omz_models/intel/asl-recognition-0004/FP16/asl-recognition-0004.xml -d CPU -hint latency
   ```

   ```sh
   [Step 1/11] Parsing and validating input arguments
   [ INFO ] Parsing input parameters
   [ INFO ] Input command: /home/openvino/bin/intel64/DEBUG/benchmark_app -m omz_models/intel/asl-recognition-0004/FP16/asl-recognition-0004.xml -d CPU -hint latency
   [Step 2/11] Loading OpenVINO Runtime
   [ INFO ] OpenVINO:
   [ INFO ] Build ................................. 2022.3.0-7750-c1109a7317e-feature/py_cpp_align
   [ INFO ]
   [ INFO ] Device info:
   [ INFO ] CPU
   [ INFO ] Build ................................. 2022.3.0-7750-c1109a7317e-feature/py_cpp_align
   [ INFO ]
   [ INFO ]
   [Step 3/11] Setting device configuration
   [ WARNING ] Device(CPU) performance hint is set to LATENCY
   [Step 4/11] Reading model files
   [ INFO ] Loading model files
   [ INFO ] Read model took 141.11 ms
   [ INFO ] Original model I/O parameters:
   [ INFO ] Network inputs:
   [ INFO ]     input (node: input) : f32 / [N,C,D,H,W] / {1,3,16,224,224}
   [ INFO ] Network outputs:
   [ INFO ]     output (node: output) : f32 / [...] / {1,100}
   [Step 5/11] Resizing model to match image sizes and given batch
   [ INFO ] Model batch size: 0
   [Step 6/11] Configuring input of the model
   [ INFO ] Model batch size: 1
   [ INFO ] Network inputs:
   [ INFO ]     input (node: input) : f32 / [N,C,D,H,W] / {1,3,16,224,224}
   [ INFO ] Network outputs:
   [ INFO ]     output (node: output) : f32 / [...] / {1,100}
   [Step 7/11] Loading the model to the device
   [ INFO ] Compile model took 989.62 ms
   [Step 8/11] Querying optimal runtime parameters
   [ INFO ] Model:
   [ INFO ]   NETWORK_NAME: torch-jit-export
   [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 2
   [ INFO ]   NUM_STREAMS: 2
   [ INFO ]   AFFINITY: CORE
   [ INFO ]   INFERENCE_NUM_THREADS: 0
   [ INFO ]   PERF_COUNT: NO
   [ INFO ]   INFERENCE_PRECISION_HINT: f32
   [ INFO ]   PERFORMANCE_HINT: LATENCY
   [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
   [Step 9/11] Creating infer requests and preparing input tensors
   [ WARNING ] No input files were given: all inputs will be filled with random values!
   [ INFO ] Test Config 0
   [ INFO ] input  ([N,C,D,H,W], f32, {1, 3, 16, 224, 224}, static):       random (binary data is expected)
   [Step 10/11] Measuring performance (Start inference asynchronously, 2 inference requests, limits: 60000 ms duration)
   [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
   [ INFO ] First inference took 37.27 ms
   [Step 11/11] Dumping statistics report
   [ INFO ] Count:        5470 iterations
   [ INFO ] Duration:     60028.56 ms
   [ INFO ] Latency:
   [ INFO ]    Median:     21.79 ms
   [ INFO ]    Average:    21.92 ms
   [ INFO ]    Min:        20.60 ms
   [ INFO ]    Max:        37.19 ms
   [ INFO ] Throughput:   91.12 FPS
   ```
The Benchmark Tool can also be used with dynamically shaped networks to measure expected inference time for various input data shapes. See the `-shape` and `-data_shape` argument descriptions in the <a href="#all-configuration-options-cpp-benchmark">All configuration options</a> section to learn more about using dynamic shapes. Here is a command example for using benchmark_app with dynamic networks and a portion of the resulting output:

   ```sh
   ./benchmark_app -m omz_models/intel/asl-recognition-0004/FP16/asl-recognition-0004.xml -d CPU -shape [-1,3,16,224,224] -data_shape [1,3,16,224,224][2,3,16,224,224][4,3,16,224,224] -pcseq
   ```

   ```sh
   [Step 9/11] Creating infer requests and preparing input tensors
   [ INFO ] Test Config 0
   [ INFO ] input  ([N,C,D,H,W], f32, {1, 3, 16, 224, 224}, dyn:{?,3,16,224,224}): random (binary data is expected)
   [ INFO ] Test Config 1
   [ INFO ] input  ([N,C,D,H,W], f32, {2, 3, 16, 224, 224}, dyn:{?,3,16,224,224}): random (binary data is expected)
   [ INFO ] Test Config 2
   [ INFO ] input  ([N,C,D,H,W], f32, {4, 3, 16, 224, 224}, dyn:{?,3,16,224,224}): random (binary data is expected)
   [Step 10/11] Measuring performance (Start inference asynchronously, 11 inference requests, limits: 60000 ms duration)
   [ INFO ] Benchmarking in full mode (inputs filling are included in measurement loop).
   [ INFO ] First inference took 204.40 ms
   [Step 11/11] Dumping statistics report
   [ INFO ] Count:        2783 iterations
   [ INFO ] Duration:     60326.29 ms
   [ INFO ] Latency:
   [ INFO ]    Median:     208.20 ms
   [ INFO ]    Average:    237.47 ms
   [ INFO ]    Min:        85.06 ms
   [ INFO ]    Max:        743.46 ms
   [ INFO ] Latency for each data shape group:
   [ INFO ] 1. input: {1, 3, 16, 224, 224}
   [ INFO ]    Median:     120.36 ms
   [ INFO ]    Average:    117.19 ms
   [ INFO ]    Min:        85.06 ms
   [ INFO ]    Max:        348.66 ms
   [ INFO ] 2. input: {2, 3, 16, 224, 224}
   [ INFO ]    Median:     207.81 ms
   [ INFO ]    Average:    206.39 ms
   [ INFO ]    Min:        167.19 ms
   [ INFO ]    Max:        578.33 ms
   [ INFO ] 3. input: {4, 3, 16, 224, 224}
   [ INFO ]    Median:     387.40 ms
   [ INFO ]    Average:    388.99 ms
   [ INFO ]    Min:        327.50 ms
   [ INFO ]    Max:        743.46 ms
   [ INFO ] Throughput:   107.61 FPS
   ```

## See Also
* [Using OpenVINO Runtime Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
* [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](@ref omz_tools_downloader)
