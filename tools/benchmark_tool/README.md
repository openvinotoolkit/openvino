# Benchmark Python* Tool {#openvino_inference_engine_tools_benchmark_tool_README}

This topic demonstrates how to run the Benchmark Python* Tool, which performs inference using convolutional networks.
Performance can be measured for two inference modes: latency- and throughput-oriented.

> **NOTE**: This topic describes usage of Python implementation of the Benchmark Tool. For the C++ implementation, refer to [Benchmark C++ Tool](../../samples/cpp/benchmark_app/README.md).

## How It Works
Upon start-up, the application reads command-line parameters and loads a network and inputs (images/binary files) to the specified device.
Device-specific execution parameters (number of streams, threads, and so on) can be either explicitly specified through the command line
or left default. In the latter case, the sample logic will select the values for the optimal throughput.
While further experimenting with individual parameters (like number of streams and requests, batch size, etc) allows to find the performance sweet spot,
usually, the resulting values are not very performance-portable,
so the values from one machine or device are not necessarily optimal for another.
From this perspective, the most portable way is experimenting only the performance hints. To learn more, refer to the section below.

> **NOTE**: By default, OpenVINO samples, tools and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model to Intermediate Representation (IR)](../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).

### Latency and Throughput-focused Inference Modes
In many cases the primary performance metric is the time (in milliseconds) for an individual inference request.
For conventional devices the best latency is usually achieved when the application operates single inference request.
Similarly, while for some devices the synchronous API (`Infer` method) was slightly better for the latency.
However, advanced devices like multi-socket CPUs, modern GPUs and so on, are capable to run multiple inference requests,
while delivering the same latency (as with the single request). Also, the asynchronous API is more general/flexible
(with respect to handling multiple inference requests).
Overall, the legacy way of measuring latency (triggered by '-api sync') with a single request and synchronous API is discouraged
in favor of the dedicated '-hint latency' that lets the _device_ to apply the right settings to minimize the time to request.

Throughput-oriented scenarios, in contrast, are focused on fully saturating the machine with enough data to crunch,
as opposite to the time of the individual request. So, the primary performance metric is rather FPS (frames per second).
Yet, just like with the latency case, the optimal execution parameters may differ between machines and devices.
So, again, as explained in the previous section, the most portable way is to use the dedicated performance hint, rather than playing individual parameters.
The hints allow the device to configure actual settings for the specified mode. The sample then queries/executes the optimal number of inference requests.

During the execution, the application collects/reports two types of metrics:
* Wall-clock time (latency) of each infer request and resulting latency
* Duration of all inference executions and resulting throughput
By default, the reported latency value is always calculated as the median (i.e. 50th percentile) value of all collected latencies from individual requests.
Notice that you can change the desired percentile with the command-line flag.
The throughput value is derived from the overall inference execution time and number of completed requests (respecting the batch size).

### Defining the Number of Inference Executions
A number of executions is defined by one of the two values:
* Explicitly, with the `-niter` command-line argument
* As _time_ duration specified with the `-t` command-line argument
* Both of them (execution will continue until both conditions are met)
* Predefined duration if neither `-niter` nor `-t` are not specified. Predefined duration value depends on the device.

## Run the Tool

Before running the Benchmark tool, install  [OpenVINO™ Development Tools](../../docs/install_guides/installing-model-dev-tools.md).

Notice that the benchmark_app usually produces optimal performance for any device out of the box.
**So in most cases you don't need to play the app options explicitly and the plain device name is enough**, for example, for CPU:

```sh
benchmark_app -m <model> -i <input> -d CPU
```

But it is still may be sub-optimal for some cases, especially for very small networks. More details can read in [Performance Optimization Guide](../../docs/optimization_guide/dldt_optimization_guide.md).

Running the application with the `-h` or `--help`' option yields the following usage message:

```
usage: benchmark_app [-h [HELP]] [-i PATHS_TO_INPUT [PATHS_TO_INPUT ...]] -m PATH_TO_MODEL 
                     [-d TARGET_DEVICE] 
                     [-l PATH_TO_EXTENSION] [-c PATH_TO_CLDNN_CONFIG] 
                     [-api {sync,async}]
                     [-niter NUMBER_ITERATIONS]
                     [-nireq NUMBER_INFER_REQUESTS]
                     [-b BATCH_SIZE]
                     [-stream_output [STREAM_OUTPUT]]
                     [-t TIME]
                     [-progress [PROGRESS]]
                     [-shape SHAPE]
                     [-layout LAYOUT]
                     [-nstreams NUMBER_STREAMS]
                     [-enforcebf16 [{True,False}]]
                     [-nthreads NUMBER_THREADS]
                     [-pin {YES,NO,NUMA,HYBRID_AWARE}]
                     [-exec_graph_path EXEC_GRAPH_PATH]
                     [-pc [PERF_COUNTS]]
                     [-report_type {no_counters,average_counters,detailed_counters}]
                     [-report_folder REPORT_FOLDER]
                     [-dump_config DUMP_CONFIG]
                     [-load_config LOAD_CONFIG]
                     [-qb {8,16}]
                     [-ip {U8,FP16,FP32}]
                     [-op {U8,FP16,FP32}]
                     [-iop INPUT_OUTPUT_PRECISION]
                     [-cdir CACHE_DIR]
                     [-lfile [LOAD_FROM_FILE]]

Options:
  -h [HELP], --help [HELP]
                        Show this help message and exit.
  -i PATHS_TO_INPUT [PATHS_TO_INPUT ...], --paths_to_input PATHS_TO_INPUT [PATHS_TO_INPUT ...]
                        Optional. Path to a folder with images and/or binaries or to specific image or binary file.
  -m PATH_TO_MODEL, --path_to_model PATH_TO_MODEL
                        Required. Path to an .xml/.onnx/.prototxt file with a trained model or to a .blob file with a trained compiled model.
  -d TARGET_DEVICE, --target_device TARGET_DEVICE
                        Optional. Specify a target device to infer on (the list of available devices is shown below). Default value is CPU. Use '-d HETERO:<comma separated devices list>' format to specify
                        HETERO plugin. Use '-d MULTI:<comma separated devices list>' format to specify MULTI plugin. The application looks for a suitable plugin for the specified device.
  -l PATH_TO_EXTENSION, --path_to_extension PATH_TO_EXTENSION
                        Optional. Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
  -c PATH_TO_CLDNN_CONFIG, --path_to_cldnn_config PATH_TO_CLDNN_CONFIG
                        Optional. Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
  -api {sync,async}, --api_type {sync,async}
                        Optional. Enable using sync/async API. Default value is async.
  -niter NUMBER_ITERATIONS, --number_iterations NUMBER_ITERATIONS
                        Optional. Number of iterations. If not specified, the number of iterations is calculated depending on a device.
  -nireq NUMBER_INFER_REQUESTS, --number_infer_requests NUMBER_INFER_REQUESTS
                        Optional. Number of infer requests. Default value is determined automatically for device.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation
  -stream_output [STREAM_OUTPUT]
                        Optional. Print progress as a plain text. When specified, an interactive progress bar is replaced with a multi-line output.
  -t TIME, --time TIME  Optional. Time in seconds to execute topology.
  -progress [PROGRESS]  Optional. Show progress bar (can affect performance measurement). Default values is 'False'.
  -shape SHAPE          Optional. Set shape for input. For example, "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]" in case of one input size.
  -layout LAYOUT        Optional. Prompts how network layouts should be treated by application. For example, "input1[NCHW],input2[NC]" or "[NCHW]" in case of one input size.
  -nstreams NUMBER_STREAMS, --number_streams NUMBER_STREAMS
                        Optional. Number of streams to use for inference on the CPU/GPU/MYRIAD (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).
                        Default value is determined automatically for a device. Please note that although the automatic selection usually provides a reasonable performance, it still may be non - optimal for
                        some cases, especially for very small networks. Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency estimations the number of streams should be
                        set to 1. See samples README for more details.
  -enforcebf16 [{True,False}], --enforce_bfloat16 [{True,False}]
                        Optional. By default floating point operations execution in bfloat16 precision are enforced if supported by platform. 'true' - enable bfloat16 regardless of platform support. 'false' -
                        disable bfloat16 regardless of platform support.
  -nthreads NUMBER_THREADS, --number_threads NUMBER_THREADS
                        Number of threads to use for inference on the CPU, GNA (including HETERO and MULTI cases).
  -pin {YES,NO,NUMA,HYBRID_AWARE}, --infer_threads_pinning {YES,NO,NUMA,HYBRID_AWARE}
                        Optional. Enable threads->cores ('YES' which is OpenVINO runtime's default for conventional CPUs), threads->(NUMA)nodes ('NUMA'), threads->appropriate core types ('HYBRID_AWARE', which
                        is OpenVINO runtime's default for Hybrid CPUs)or completely disable ('NO')CPU threads pinning for CPU-involved inference.
  -exec_graph_path EXEC_GRAPH_PATH, --exec_graph_path EXEC_GRAPH_PATH
                        Optional. Path to a file where to store executable graph information serialized.
  -pc [PERF_COUNTS], --perf_counts [PERF_COUNTS]
                        Optional. Report performance counters.
  -report_type {no_counters,average_counters,detailed_counters}, --report_type {no_counters,average_counters,detailed_counters}
                        Optional. Enable collecting statistics report. "no_counters" report contains configuration options specified, resulting FPS and latency. "average_counters" report extends "no_counters"
                        report and additionally includes average PM counters values for each layer from the network. "detailed_counters" report extends "average_counters" report and additionally includes per-
                        layer PM counters and latency for each executed infer request.
  -report_folder REPORT_FOLDER, --report_folder REPORT_FOLDER
                        Optional. Path to a folder where statistics report is stored.
  -dump_config DUMP_CONFIG
                        Optional. Path to JSON file to dump IE parameters, which were set by application.
  -load_config LOAD_CONFIG
                        Optional. Path to JSON file to load custom IE parameters. Please note, command line parameters have higher priority then parameters from configuration file.
  -qb {8,16}, --quantization_bits {8,16}
                        Optional. Weight bits for quantization: 8 (I8) or 16 (I16)
  -ip {U8,FP16,FP32}, --input_precision {U8,FP16,FP32}
                        Optional. Specifies precision for all input layers of the network.
  -op {U8,FP16,FP32}, --output_precision {U8,FP16,FP32}
                        Optional. Specifies precision for all output layers of the network.
  -iop INPUT_OUTPUT_PRECISION, --input_output_precision INPUT_OUTPUT_PRECISION
                        Optional. Specifies precision for input and output layers by name. Example: -iop "input:FP16, output:FP16". Notice that quotes are required. Overwrites precision from ip and op options
                        for specified layers.
  -cdir CACHE_DIR, --cache_dir CACHE_DIR
                        Optional. Enable model caching to specified directory
  -lfile [LOAD_FROM_FILE], --load_from_file [LOAD_FROM_FILE]
                        Optional. Loads model from file directly without read_network.
  -dopt [DENORMALS_OPTIMIZATION], --denormals_optimization [DENORMALS_OPTIMIZATION]
                        Optional. Denormals is optimized by treating as zeros.
  -cpu_experimental SETTING     Optional. Enable experimental setting for CPU plugin. SETTING should be "brgconv" etc.
```
Running the application with the empty list of options yields the usage message given above and an error message.

Application supports topologies with one or more inputs. If a topology is not data sensitive, you can skip the input parameter. In this case, inputs are filled with random values.
If a model has only image input(s), please a provide folder with images or a path to an image as input.
If a model has some specific input(s) (not images), please prepare a binary file(s), which is filled with data of appropriate precision and provide a path to them as input.
If a model has mixed input types, input folder should contain all required files. Image inputs are filled with image files one by one. Binary inputs are filled with binary inputs one by one.

To run the tool, you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).

> **NOTE**: Before running the tool with a trained model, make sure the model is converted to the OpenVINO format (\*.xml + \*.bin) using the [Model Optimizer tool](../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

## Examples of Running the Tool

This section provides step-by-step instructions on how to run the Benchmark Tool with the `googlenet-v1` public model on CPU or GPU devices. The [dog.bmp](https://storage.openvinotoolkit.org/data/test_data/images/224x224/dog.bmp) file is used as an input.

> **NOTE**: The Internet access is required to execute the following steps successfully. If you have access to the Internet through the proxy server only, please make sure that it is configured in your OS environment.

1. Install OpenVINO Development Tools to work with Caffe* models:

   ``` sh
   pip install openvino-dev[caffe]
   ```

2. Download the model. Go to the Model Downloader directory and run the `omz_downloader` tool with the model name and directory to download the model to:

   ```sh
   omz_downloader --name googlenet-v1 -o <models_dir>
   ```
3. Convert the model to the OpenVINO IR format. Run Model Optimizer with the path to the model, model format and output directory to generate the IR files:
   ```sh
   mo --input_model <models_dir>/public/googlenet-v1/googlenet-v1.caffemodel --data_type FP32 --output_dir <ir_dir>
   ```
4. Run the tool with specifying the `dog.bmp` file as an input image, the IR of the `googlenet-v1` model and a device to perform inference on. The following commands demonstrate running the Benchmark Tool in the asynchronous mode on CPU and GPU devices:

   * On CPU:
   ```sh
    benchmark_app -m <ir_dir>/googlenet-v1.xml -d CPU -api async -i dog.bmp -progress -b 1
   ```
   * On GPU:
   ```sh
   benchmark_app -m <ir_dir>/googlenet-v1.xml -d GPU -api async -i dog.bmp -progress -b 1
   ```

The application outputs number of executed iterations, total duration of execution, latency and throughput.
Additionally, if you set the `-pc` parameter, the application outputs performance counters.
If you set `-exec_graph_path`, the application reports executable graph information serialized.

Below are fragments of sample output for static and dynamic models:
* For static model:
   ```
   [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests using 4 streams for CPU, inference only: True, limits: 60000 ms duration)
   [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
   [ INFO ] First inference took 5.00 ms
   [Step 11/11] Dumping statistics report
   Count:          29936 iterations
   Duration:       60010.13 ms
   Latency:
      Median:     7.30 ms
      AVG:        7.97 ms
      MIN:        5.02 ms
      MAX:        29.26 ms
   Throughput: 498.85 FPS
   ```
* For dynamic model:
   ```
   [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests using 4 streams for CPU, inference only: False, limits: 60000 ms duration)
   [ INFO ] Benchmarking in full mode (inputs filling are included in measurement loop).
   [ INFO ] First inference took 5.10 ms
   [Step 11/11] Dumping statistics report
   Count:          13596 iterations
   Duration:       60028.12 ms
   Latency:
      AVG:        17.53 ms
      MIN:        2.88 ms
      MAX:        63.54 ms
   Latency for each data shape group:
   data: {1, 3, 128, 128}
      AVG:        5.09 ms
      MIN:        2.88 ms
      MAX:        23.30 ms
   data: {1, 3, 224, 224}
      AVG:        10.67 ms
      MIN:        5.97 ms
      MAX:        31.79 ms
   data: {1, 3, 448, 448}
      AVG:        36.84 ms
      MIN:        24.76 ms
      MAX:        63.54 ms
   Throughput: 226.49 FPS
   ```

## See Also
* [Using OpenVINO Samples](../../docs/OV_Runtime_UG/Samples_Overview.md)
* [Model Optimizer](../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](@ref omz_tools_downloader)
