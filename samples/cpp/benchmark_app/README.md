# Benchmark C++ Tool {#openvino_inference_engine_samples_benchmark_app_README}

This page demonstrates how to use the Benchmark C++ Tool to estimate deep learning inference performance on supported devices.

> **NOTE**: This page describes usage of the C++ implementation of the Benchmark Tool. For the Python implementation, refer to the [Benchmark Python Tool](../../../tools/benchmark_tool/README.md) page. The Python version is recommended for benchmarking models that will be used in Python applications, and the C++ version is recommended for benchmarking models that will be used in C++ applications. Both tools have a similar command interface and backend.
		

## Basic Usage
To use the C++ benchmark_app, you must first build it following the [Build the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) instructions and then set up paths and environment variables by following the [Get Ready for Running the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) instructions. Navigate to the directory where the benchmark_app C++ sample binary was built.

> **NOTE**: If you installed OpenVINO Runtime using PyPI or Anaconda Cloud, only the [Benchmark Python Tool](../../../tools/benchmark_tool/README.md) is available, and you should follow the usage instructions on that page instead.

The benchmarking application works with OpenVINO models in IR format (`model.xml` and `model.bin`) or ONNX format (`model.onnx`). You can use it with your own models or pre-trained ones. You can choose among both [public](@ref omz_models_group_public) and [Intel](@ref omz_models_group_intel) models from the Open Model Zoo, which you can download using [Model Downloader](@ref omz_tools_downloader). Before running the tool, make sure the model is either converted to the OpenVINO format, using [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md), or is in the ONNX format.

To run benchmarking with default options on a model, use the following command:

```
./benchmark_app -m model.xml
```

By default, the application will load the specified model onto the CPU and perform inferencing on batches of randomly-generated data inputs for 60 seconds. As it loads, it prints information about benchmark parameters. When benchmarking is completed, it reports the minimum, average, and maximum inferencing latency and average the throughput.

You may be able to improve benchmark results beyond the default configuration by configuring some of the execution parameters for your model. For example, you can use "throughput" or "latency" performance hints to optimize the runtime for higher FPS or reduced inferencing time. Read on to learn more about the configuration options available with benchmark_app.

## Configuration Options
The benchmark app provides various options for configuring execution parameters. This section covers key configuration options for easily tuning benchmarking to achieve better performance on your device. A list of all configuration options is given in the [Advanced Usage](#advanced-usage) section.

### Performance hints: latency and throughput
The benchmark app allows users to provide high-level "performance hints" for setting latency-focused or throughput-focused inference modes. This hint causes the runtime to automatically adjust runtime parameters, such as the number of processing streams and inference batch size, to prioritize for reduced latency or high throughput.

The performance hints do not require any device-specific settings and they are completely portable between devices. Parameters are automatically configured based on whichever device is being used. This allows users to easily port applications between hardware targets without having to re-determine the best runtime parameters for the new device.

The performance hint is set by using `-hint latency` or `-hint throughput` when running benchmark_app:

```
./benchmark_app -m model.xml -hint latency
./benchmark_app -m model.xml -hint throughput
```

#### Latency
Latency is the amount of time it takes to process a single inference request. In applications where data needs to be inferenced and acted on as quickly as possible (such as autonomous driving), low latency is desirable. For conventional devices, lower latency is achieved by reducing the amount of parallel processing streams so the system can utilize as many resources as possible to quickly calculate each inference request. However, advanced devices like multi-socket CPUs and modern GPUs are capable of running multiple inference requests while delivering the same latency.

When benchmark_app is run with `-hint latency`, it determines the optimal number of parallel inference requests for minimizing latency while still maximizing the parallelization capabilities of the hardware. It automatically sets the number of processing streams and inference batch size to achieve the best latency.

#### Throughput
Throughput is the amount of data an inferencing pipeline can process at once, and it is usually measured in frames per second (FPS) or inferences per second. In applications where large amounts of data needs to be inferenced simultaneously (such as multi-camera video streams), high throughput is needed. To achieve high throughput, the runtime focuses on fully saturating the device with enough data to process. It utilizes as much memory and as many parallel streams as possible to maximize the amount of data that can be processed simultaneously.

When benchmark_app is run with `-hint throughput`, it automatically sets the inference batch size to fill up all the memory available. It also maximizes the number of parallel inference requests to utilize all the threads available on the device.

For more information on performance hints, see the [High-level Performance Hints](../../../docs/OV_Runtime_UG/performance_hints.md) page. For more details on optimal runtime configurations and how they are automatically determined using performance hints, see [Runtime Inference Optimizations](../../../docs/optimization_guide/dldt_deployment_optimization_guide.md).


### Device
To set which device benchmarking runs on, use the `-d <device>` argument. This will tell benchmark_app to run benchmarking on that specific device. The benchmark app supports "CPU", "GPU", and "MYRIAD" (also known as [VPU](../../../docs/OV_Runtime_UG/supported_plugins/VPU.md)) devices. In order to use the GPU or VPU, the system must have the appropriate drivers installed. If no device is specified, benchmark_app will default to using CPU.

For example, to run benchmarking on GPU, use:

```
./benchmark_app -m model.xml -d GPU
```

You may also specify "AUTO" as the device, and benchmark_app will automatically select the best device to run benchmarking on. For more information, see the [Automatic device selection](../../../docs/OV_Runtime_UG/auto_device_selection.md) page.

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
For more usage examples (and step-by-step instructions on how to set up a model for benchmarking), see the [Examples of Running the Tool](#examples-of-running-the-tool) section.

## Advanced Usage

> **NOTE**: By default, OpenVINO samples, tools and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channel order in the sample or demo application or reconvert your model using the Model Optimizer tool with --reverse_input_channels argument specified. For more information about the argument, refer to When to Reverse Input Channels section of Converting a Model to Intermediate Representation (IR).

### Per-layer performance and logging
The application also collects per-layer Performance Measurement (PM) counters for each executed infer request if you enable statistics dumping by setting the `-report_type` parameter to one of the possible values:

* `no_counters` report includes configuration options specified, resulting FPS and latency.
* `average_counters` report extends the `no_counters` report and additionally includes average PM counters values for each layer from the network.
* `detailed_counters` report extends the `average_counters` report and additionally includes per-layer PM counters and latency for each executed infer request.

Depending on the type, the report is stored to benchmark_no_counters_report.csv, benchmark_average_counters_report.csv, or benchmark_detailed_counters_report.csv file located in the path specified in -report_folder. The application also saves executable graph information serialized to an XML file if you specify a path to it with the -exec_graph_path parameter.

### All configuration options

Running the application with the `-h` or `--help` option yields the following usage message:

```
./benchmark_app -h

benchmark_app [OPTION]
Options:

    -h, --help                Print a usage message
    -m "<path>"               Required. Path to an .xml/.onnx file with a trained model or to a .blob files with a trained compiled model.
    -i "<path>"               Optional. Path to a folder with images and/or binaries or to specific image or binary file.
                              In case of dynamic shapes networks with several inputs provide the same number of files for each input (except cases with single file for any input):"input1:1.jpg input2:1.bin", "input1:1.bin,2.bin input2:3.bin input3:4.bin,5.bin ". Also you can pass specific keys for inputs: "random" - for fillling input with random data, "image_info" - for filling input with image size.
                              You should specify either one files set to be used for all inputs (without providing input names) or separate files sets for every input of model (providing inputs names).
    -d "<device>"             Optional. Specify a target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. Use "-d MULTI:<comma-separated_devices_list>" format to specify MULTI plugin. The application looks for a suitable plugin for the specified device.
    -l "<absolute_path>"      Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
          Or
    -c "<absolute_path>"      Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
    -hint "performance hint (latency or throughput or none)"   Optional. Performance hint allows the OpenVINO device to select the right network-specific settings.
                               'throughput' or 'tput': device performance mode will be set to THROUGHPUT.
                               'latency': device performance mode will be set to LATENCY.
                               'none': no device performance mode will be set.
                              Using explicit 'nstreams' or other device-specific options, please set hint to 'none'
    -api "<sync/async>"       Optional (deprecated). Enable Sync/Async API. Default value is "async".
    -niter "<integer>"        Optional. Number of iterations. If not specified, the number of iterations is calculated depending on a device.
    -nireq "<integer>"        Optional. Number of infer requests. Default value is determined automatically for device.
    -b "<integer>"            Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation.
    -stream_output            Optional. Print progress as a plain text. When specified, an interactive progress bar is replaced with a multiline output.
    -t                        Optional. Time in seconds to execute topology.
    -progress                 Optional. Show progress bar (can affect performance measurement). Default values is "false".
    -shape                    Optional. Set shape for network input. For example, "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]" in case of one input size. This parameter affect model input shape and can be dynamic. For dynamic dimensions use symbol `?` or '-1'. Ex. [?,3,?,?]. For bounded dimensions specify range 'min..max'. Ex. [1..10,3,?,?].
    -data_shape               Required for networks with dynamic shapes. Set shape for input blobs. In case of one input size: "[1,3,224,224]" or "input1[1,3,224,224],input2[1,4]". In case of several input sizes provide the same number for each input (except cases with single shape for any input): "[1,3,128,128][3,3,128,128][1,3,320,320]", "input1[1,1,128,128][1,1,256,256],input2[80,1]" or "input1[1,192][1,384],input2[1,192][1,384],input3[1,192][1,384],input4[1,192][1,384]". If network shapes are all static specifying the option will cause an exception.
    -layout                   Optional. Prompts how network layouts should be treated by application. For example, "input1[NCHW],input2[NC]" or "[NCHW]" in case of one input size.
    -cache_dir "<path>"       Optional. Enables caching of loaded models to specified directory. List of devices which support caching is shown at the end of this message.
    -load_from_file           Optional. Loads model from file directly without ReadNetwork. All CNNNetwork options (like re-shape) will be ignored
    -latency_percentile       Optional. Defines the percentile to be reported in latency metric. The valid range is [1, 100]. The default value is 50 (median).

  Device-specific performance options:
    -nstreams "<integer>"     Optional. Number of streams to use for inference on the CPU, GPU or MYRIAD devices (for HETERO and MULTI device cases use format <dev1>:<nstreams1>,<dev2>:<nstreams2> or just <nstreams>). Default value is determined automatically for a device.Please note that although the automatic selection usually provides a reasonable performance, it still may be non - optimal for some cases, especially for very small networks. See sample's README for more details. Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency estimations the number of streams should be set to 1.
    -nthreads "<integer>"     Optional. Number of threads to use for inference on the CPU (including HETERO and MULTI cases).
    -pin ("YES"|"CORE")/"HYBRID_AWARE"/("NO"|"NONE")/"NUMA"   Optional. Explicit inference threads binding options (leave empty to let the OpenVINO to make a choice):
                                enabling threads->cores pinning("YES", which is already default for any conventional CPU),
                                letting the runtime to decide on the threads->different core types("HYBRID_AWARE", which is default on the hybrid CPUs)
                                threads->(NUMA)nodes("NUMA") or
                                completely disable("NO") CPU inference threads pinning

  Statistics dumping options:
    -report_type "<type>"       Optional. Enable collecting statistics report. "no_counters" report contains configuration options specified, resulting FPS and latency.
                                "average_counters" report extends "no_counters" report and additionally includes average PM counters values for each layer from the network.
                                "detailed_counters" report extends "average_counters" report and additionally includes per-layer PM counters
                                and latency for each executed infer request.
    -report_folder              Optional. Path to a folder where statistics report is stored.
    -exec_graph_path            Optional. Path to a file where to store executable graph information serialized.
    -pc                         Optional. Report performance counters.
    -dump_config                Optional. Path to JSON file to dump IE parameters, which were set by application.
    -load_config                Optional. Path to JSON file to load custom IE parameters. Please note, command line parameters have higher priority than parameters from configuration file.

   Statistics dumping options:
    -report_type "<type>"     Optional. Enable collecting statistics report. "no_counters" report contains configuration options specified, resulting FPS and latency. "average_counters" report extends "no_counters" report and additionally includes average PM counters values for each layer from the network. "detailed_counters" report extends "average_counters" report and additionally includes per-layer PM counters and latency for each executed infer request.
    -report_folder            Optional. Path to a folder where statistics report is stored.
    -json_stats               Optional. Enables JSON-based statistics output (by default reporting system will use CSV format). Should be used together with -report_folder option.    -exec_graph_path          Optional. Path to a file where to store executable graph information serialized.
    -pc                       Optional. Report performance counters.
    -pcseq                    Optional. Report latencies for each shape in -data_shape sequence.
    -dump_config              Optional. Path to JSON file to dump IE parameters, which were set by application.
    -load_config              Optional. Path to JSON file to load custom IE parameters. Please note, command line parameters have higher priority then parameters from configuration file.
    -infer_precision "<element type>"Optional. Inference precission
    -ip                          <value>     Optional. Specifies precision for all input layers of the network.
    -op                          <value>     Optional. Specifies precision for all output layers of the network.
    -iop                        "<value>"    Optional. Specifies precision for input and output layers by name.
                                             Example: -iop "input:FP16, output:FP16".
                                             Notice that quotes are required.
                                             Overwrites precision from ip and op options for specified layers.
    -iscale                    Optional. Scale values to be used for the input image per channel.
Values to be provided in the [R, G, B] format. Can be defined for desired input of the model.
Example: -iscale data[255,255,255],info[255,255,255]

    -imean                     Optional. Mean values to be used for the input image per channel.
Values to be provided in the [R, G, B] format. Can be defined for desired input of the model,
Example: -imean data[255,255,255],info[255,255,255]

    -inference_only              Optional. Measure only inference stage. Default option for static models. Dynamic models are measured in full mode which includes inputs setup stage, inference only mode available for them with single input data shape only. To enable full mode for static models pass "false" value to this argument: ex. "-inference_only=false".
```

Running the application with the empty list of options yields the usage message given above and an error message.

### More information on inputs
The benchmark tool supports topologies with one or more inputs. If a topology is not data sensitive, you can skip the input parameter, and the inputs will be filled with random values. If a model has only image input(s), provide a folder with images or a path to an image as input. If a model has some specific input(s) (besides images), please prepare a binary file(s) that is filled with data of appropriate precision and provide a path to it as input. If a model has mixed input types, the input folder should contain all required files. Image inputs are filled with image files one by one. Binary inputs are filled with binary inputs one by one.

## Examples of Running the Tool
This section provides step-by-step instructions on how to run the Benchmark Tool with the `googlenet-v1` public model on CPU or GPU devices.  The [dog.bmp](https://storage.openvinotoolkit.org/data/test_data/images/224x224/dog.bmp) file is used as an input.

> **NOTE**: Internet access is required to execute the following steps successfully. If you have access to the Internet through a proxy server only, please make sure that it is configured in your OS environment.

1. Install OpenVINO Development Tools to work with Caffe* models:

   ``` sh
   pip install openvino-dev[caffe]
   ```

2. Download the model. Go to the Model Downloader directory and run the `omz_downloader` script with specifying the model name and directory to download the model to:

   ```sh
   omz_downloader --name googlenet-v1 -o <models_dir>
   ```

3. Convert the model to the OpenVINO IR format. Run the Model Optimizer using the `mo` command with the path to the model, model format and output directory to generate the IR files:

   ```sh
   mo --input_model <models_dir>/public/googlenet-v1/googlenet-v1.caffemodel --data_type FP32 --output_dir <ir_dir>
   ```

4. Run the tool specifying the `dog.bmp` file as an input image, the IR of the `googlenet-v1` model, and a device to perform inference on. The following commands demonstrate running the Benchmark Tool in the throughput mode on CPU and GPU devices:

   * On CPU:
   ```sh
   ./benchmark_app -m <ir_dir>/googlenet-v1.xml -i dog.bmp  -d CPU -hint throughput -progress
   ```
   * On GPU:
   ```sh
   ./benchmark_app -m <ir_dir>/googlenet-v1.xml -i dog.bmp -d GPU -hint throughput -progress
   ```

The application outputs the number of executed iterations, total duration of execution, latency, and throughput.
Additionally, if you set the `-report_type` parameter, the application outputs statistics report. If you set the `-pc` parameter, the application outputs performance counters. If you set `-exec_graph_path`, the application reports executable graph information serialized. All measurements including per-layer PM counters are reported in milliseconds.

Below are fragments of sample output static and dynamic networks:

* For static network:
   ```
   [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests using 4 streams for CPU, limits: 60000 ms duration)
   [ INFO ] BENCHMARK IS IN INFERENCE ONLY MODE.
   [ INFO ] Input blobs will be filled once before performance measurements.
   [ INFO ] First inference took 26.26 ms
   Progress: [................... ]  99% done

   [Step 11/11] Dumping statistics report
   [ INFO ] Count:      6640 iterations
   [ INFO ] Duration:   60039.70 ms
   [ INFO ] Latency:
   [ INFO ]        Median:  35.36 ms
   [ INFO ]        Avg:    36.12 ms
   [ INFO ]        Min:    18.55 ms
   [ INFO ]        Max:    88.96 ms
   [ INFO ] Throughput: 110.59 FPS
   ```

* For dynamic network:
   ```
   [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests using 4 streams for CPU, limits: 60000 ms duration)
   [ INFO ] BENCHMARK IS IN FULL MODE.
   [ INFO ] Inputs setup stage will be included in performance measurements.
   [ INFO ] First inference took 26.80 ms
   Progress: [................... ]  99% done

   [Step 11/11] Dumping statistics report
   [ INFO ] Count:      5199 iterations
   [ INFO ] Duration:   60043.34 ms
   [ INFO ] Latency:
   [ INFO ]        Median:  41.58 ms
   [ INFO ]        Avg:    46.07 ms
   [ INFO ]        Min:    8.44 ms
   [ INFO ]        Max:    115.65 ms
   [ INFO ] Latency for each data shape group:
   [ INFO ] 1. data : [1, 3, 224, 224]
   [ INFO ]        Median:  38.37 ms
   [ INFO ]        Avg:    30.29 ms
   [ INFO ]        Min:    8.44 ms
   [ INFO ]        Max:    61.30 ms
   [ INFO ] 2. data : [1, 3, 448, 448]
   [ INFO ]        Median:  68.21 ms
   [ INFO ]        Avg:    61.85 ms
   [ INFO ]        Min:    29.58 ms
   [ INFO ]        Max:    115.65 ms
   [ INFO ] Throughput: 86.59 FPS
   ```

## See Also
* [Using OpenVINO Runtime Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
* [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](@ref omz_tools_downloader)
