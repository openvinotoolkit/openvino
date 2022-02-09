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

> **NOTE**: By default, Inference Engine samples, tools and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model to Intermediate Representation (IR)](../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).

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
* Predefined duration if neither `-niter`nor `-t` are not specified. Predefined duration value depends on the device.

## Run the Tool

Before running the Benchmark tool, install the requirements:
```sh
pip install -r  requirements.txt
```

Notice that the benchmark_app usually produces optimal performance for any device out of the box.

**So in most cases you don't need to play the app options explicitly and the plain device name is enough**, for example, for CPU:
```sh
python3 benchmark_app.py -m <model> -i <input> -d CPU
```

But it is still may be sub-optimal for some cases, especially for very small networks. More details can read in [Performance Optimization Guide](../../docs/optimization_guide/dldt_optimization_guide.md).

Running the application with the `-h` or `--help`' option yields the following usage message:

```
usage: benchmark_app.py [-h] [-i PATH_TO_INPUT] -m PATH_TO_MODEL
                        [-d TARGET_DEVICE]
                        [-l PATH_TO_EXTENSION] [-c PATH_TO_CLDNN_CONFIG]
                        [-hint {throughput, latency}]
                        [-api {sync,async}] [-niter NUMBER_ITERATIONS]
                        [-b BATCH_SIZE]
                        [-stream_output [STREAM_OUTPUT]] [-t TIME]
                        [-progress [PROGRESS]] [-nstreams NUMBER_STREAMS]
                        [-nthreads NUMBER_THREADS] [-pin {YES,NO,NUMA,HYBRID_AWARE}]
                        [--exec_graph_path EXEC_GRAPH_PATH]
                        [-pc [PERF_COUNTS]]

Options:
  -h, --help            Show this help message and exit.
  -i PATH_TO_INPUT, --path_to_input PATH_TO_INPUT
                        Optional. Path to a folder with images and/or binaries
                        or to specific image or binary file.
  -m PATH_TO_MODEL, --path_to_model PATH_TO_MODEL
                        Required. Path to an .xml/.onnx/.prototxt file with a
                        trained model or to a .blob file with a trained
                        compiled model.
  -d TARGET_DEVICE, --target_device TARGET_DEVICE
                        Optional. Specify a target device to infer on: CPU,
                        GPU, HDDL or MYRIAD.
                        Use "-d HETERO:<comma separated devices list>" format to specify HETERO plugin.
                        Use "-d MULTI:<comma separated devices list>" format to specify MULTI plugin.
                        The application looks for a suitable plugin for the specified device.
  -l PATH_TO_EXTENSION, --path_to_extension PATH_TO_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        path to a shared library with the kernels
                        implementations.
  -c PATH_TO_CLDNN_CONFIG, --path_to_cldnn_config PATH_TO_CLDNN_CONFIG
                        Optional. Required for GPU custom kernels. Absolute
                        path to an .xml file with the kernels description.
  -hint {throughput, latency}, --perf_hint {throughput, latency}
                        Optional. Performance hint (optimize for latency or throughput).
                        The hint allows the OpenVINO device to select the right network-specific settings,
                        as opposite to defining specific values like  \nstreams\ from the command line.
                        So you can specify just the hint without adding explicit device-specific options.
  -api {sync,async}, --api_type {sync,async}
                        Optional. Enable using sync/async API. Default value
                        is async.
  -niter NUMBER_ITERATIONS, --number_iterations NUMBER_ITERATIONS
                        Optional. Number of iterations. If not specified, the
                        number of iterations is calculated depending on a
                        device.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Optional. Batch size value. If not specified, the
                        batch size value is determined from IR
  -stream_output [STREAM_OUTPUT]
                        Optional. Print progress as a plain text. When
                        specified, an interactive progress bar is replaced
                        with a multiline output.
  -t TIME, --time TIME  Optional. Time in seconds to execute topology.
  -progress [PROGRESS]  Optional. Show progress bar (can affect performance
                        measurement). Default values is "False".
  -shape SHAPE          Optional. Set shape for input. For example,
                        "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]"
                        in case of one input size.
  -layout LAYOUT        Optional. Prompts how network layouts should be
                        treated by application. For example,
                        "input1[NCHW],input2[NC]" or "[NCHW]" in case of one
                        input size.
  -nstreams NUMBER_STREAMS, --number_streams NUMBER_STREAMS
                       Optional. Number of streams to use for inference on the CPU/GPU/MYX in throughput mode
                       (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).
                       Default value is determined automatically for a device.
                       Please note that although the automatic selection usually provides a reasonable performance,
                       it still may be non-optimal for some cases, especially for very small networks.
  -nthreads NUMBER_THREADS, --number_threads NUMBER_THREADS
                        Number of threads to use for inference on the CPU
                        (including HETERO  and MULTI cases).
  -pin {YES,NO,NUMA,HYBRID_AWARE}, --infer_threads_pinning {YES,NO,NUMA,HYBRID_AWARE}
                        Optional. Enable threads->cores ('YES' which is OpenVINO runtime's default for conventional CPUs),
                        threads->(NUMA)nodes ('NUMA'),
                        threads->appropriate core types ('HYBRID_AWARE', which is OpenVINO runtime's default for Hybrid CPUs)
                        or completely disable ('NO')
                        CPU threads pinning for CPU-involved inference.
  --exec_graph_path EXEC_GRAPH_PATH
                        Optional. Path to a file where to store executable
                        graph information serialized.
  -pc [PERF_COUNTS], --perf_counts [PERF_COUNTS]
                        Optional. Report performance counters.
  -ip "U8"/"FP16"/"FP32"    Optional. Specifies precision for all input layers of the network.
  -op "U8"/"FP16"/"FP32"    Optional. Specifies precision for all output layers of the network.
  -iop                      Optional. Specifies precision for input and output layers by name. Example: -iop "input:FP16, output:FP16". Notice that quotes are required. Overwrites precision from ip and op options for specified layers.
```

Running the application with the empty list of options yields the usage message given above and an error message.

Application supports topologies with one or more inputs. If a topology is not data sensitive, you can skip the input parameter. In this case, inputs are filled with random values.
If a model has only image input(s), please a provide folder with images or a path to an image as input.
If a model has some specific input(s) (not images), please prepare a binary file(s), which is filled with data of appropriate precision and provide a path to them as input.
If a model has mixed input types, input folder should contain all required files. Image inputs are filled with image files one by one. Binary inputs are filled with binary inputs one by one.

To run the tool, you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).

> **NOTE**: Before running the tool with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

## Examples of Running the Tool

This section provides step-by-step instructions on how to run the Benchmark Tool with the `googlenet-v1` public model on CPU or GPU devices. As an input, the `car.png` file from the `<INSTALL_DIR>/samples/scripts/` directory is used.

> **NOTE**: The Internet access is required to execute the following steps successfully. If you have access to the Internet through the proxy server only, please make sure that it is configured in your OS environment.

1. Download the model. Go to the the Model Downloader directory and run the `downloader.py` script with the model name and directory to download the model to:
   ```sh
   cd <INSTALL_DIR>/extras/open_model_zoo/tools/downloader
   ```
   ```sh
   python3 downloader.py --name googlenet-v1 -o <models_dir>
   ```
2. Convert the model to the Inference Engine IR format. Run Model Optimizer with the path to the model, model format (which must be FP32 for CPU and FPG) and output directory to generate the IR files:
   ```sh
   mo --input_model <models_dir>/public/googlenet-v1/googlenet-v1.caffemodel --data_type FP32 --output_dir <ir_dir>
   ```
3. Run the tool with specifying the `<INSTALL_DIR>/samples/scripts/car.png` file as an input image, the IR of the `googlenet-v1` model and a device to perform inference on. The following commands demonstrate running the Benchmark Tool in the asynchronous mode on CPU and GPU devices:

   * On CPU:
   ```sh
    python3 benchmark_app.py -m <ir_dir>/googlenet-v1.xml -d CPU -api async -i <INSTALL_DIR>/samples/scripts/car.png --progress true -b 1
   ```
   * On GPU:
   ```sh
   python3 benchmark_app.py -m <ir_dir>/googlenet-v1.xml -d GPU -api async -i <INSTALL_DIR>/samples/scripts/car.png --progress true -b 1
   ```

The application outputs number of executed iterations, total duration of execution, latency and throughput.
Additionally, if you set the `-pc` parameter, the application outputs performance counters.
If you set `-exec_graph_path`, the application reports executable graph information serialized.

Below are fragments of sample output for CPU and GPU devices:
* For CPU:
   ```
   [Step 8/9] Measuring performance (Start inference asynchronously, 60000 ms duration, 4 inference requests in parallel using 4 streams)
   Progress: |................................| 100.00%

   [Step 9/9] Dumping statistics report
   Progress: |................................| 100.00%

   Count:      4408 iterations
   Duration:   60153.52 ms
   Latency:    51.8244 ms
   Throughput: 73.28 FPS
   ```
* For GPU:
   ```
   [Step 10/11] Measuring performance (Start inference asynchronously, 5 inference requests using 1 streams for CPU, limits: 120000 ms duration)
   Progress: |................................| 100%

   [Step 11/11] Dumping statistics report
   Count:      98075 iterations
   Duration:   120011.03 ms
   Latency:    5.65 ms
   Throughput: 817.22 FPS
   ```

## See Also
* [Using Inference Engine Samples](../../docs/OV_Runtime_UG/Samples_Overview.md)
* [Model Optimizer](../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](@ref omz_tools_downloader)
