# Benchmark Python* Application

This topic demonstrates how to run the Benchmark Application demo, which performs inference using convolutional networks.

## How It Works

Upon start-up, the application reads command-line parameters and loads a network and images/binary files to the Inference Engine
plugin, which is chosen depending on a specified device. The number of infer requests and execution approach depend
on the mode defined with the `-api` command-line parameter.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

### Synchronous API

For synchronous mode, the primary metric is latency. The application creates one infer request and executes the `Infer` method. A number of executions is defined by one of the two values:
* Number of iterations defined with the `-niter` command-line argument
* Time duration specified with the `-t` command-line argument
* Both of them (execution will continue until both conditions are met)
* Predefined duration if `-niter` and `-t` are not specified. Predefined duration value depends on device.

During the execution, the application collects two types of metrics:
* Latency for each infer request executed with `Infer` method
* Duration of all executions

Reported latency value is calculated as mean value of all collected latencies. Reported throughput value is a derivative from reported latency and additionally depends on batch size.

### Asynchronous API
For asynchronous mode, the primary metric is throughput in frames per second (FPS). The application creates a certain number of infer requests and executes the `StartAsync` method. A number of executions is defined by one of the two values:
* Number of iterations defined with the `-niter` command-line argument
* Time duration specified with the `-t` command-line argument
* Both of them (execution will continue until both conditions are met)
* Predefined duration if `-niter` and `-t` are not specified. Predefined duration value depends on device.

The infer requests are executed asynchronously. Callback is used to wait for previous execution to complete. The application measures all infer requests executions and reports the throughput metric based on batch size and total execution duration.

## Running

Before running the Benchmark tool, install the requirements:
```sh
pip install -r  requirements.txt
```

Notice that the benchmark_app usually produces optimal performance for any device out of the box.

**So in most cases you don't need to play the app options explicitly and the plain device name is enough**, e.g.:
```
$benchmark_app -m <model> -i <input> -d CPU
```

But it is still may be non-optimal for some cases, especially for very small networks. More details can read in [Introduction to Performance Topics](./docs/IE_DG/Intro_to_Performance.md).

Running the application with the `-h` or `--help`' option yields the following usage message:

```
usage: benchmark_app.py [-h] [-i PATH_TO_INPUT] -m PATH_TO_MODEL
                        [-d TARGET_DEVICE]
                        [-l PATH_TO_EXTENSION] [-c PATH_TO_CLDNN_CONFIG]
                        [-api {sync,async}] [-niter NUMBER_ITERATIONS]
                        [-b BATCH_SIZE]
                        [-stream_output [STREAM_OUTPUT]] [-t TIME]
                        [-progress [PROGRESS]] [-nstreams NUMBER_STREAMS]
                        [-nthreads NUMBER_THREADS] [-pin {YES,NO}]
                        [--exec_graph_path EXEC_GRAPH_PATH]
                        [-pc [PERF_COUNTS]]

Options:
  -h, --help            Show this help message and exit.
  -i PATHS_TO_INPUT [PATHS_TO_INPUT ...], --paths_to_input PATHS_TO_INPUT [PATHS_TO_INPUT ...]
                        Optional. Path to a folder with images and/or binaries
                        or to specific image or binary file.
  -m PATH_TO_MODEL, --path_to_model PATH_TO_MODEL
                        Required. Path to an .xml file with a trained model.
  -d TARGET_DEVICE, --target_device TARGET_DEVICE
                        Optional. Specify a target device to infer on (the
                        list of available devices is shown below). Default
                        value is CPU. Use '-d HETERO:<comma separated devices
                        list>' format to specify HETERO plugin. Use '-d
                        MULTI:<comma separated devices list>' format to
                        specify MULTI plugin. The application looks for a
                        suitable plugin for the specified device.
  -l PATH_TO_EXTENSION, --path_to_extension PATH_TO_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        path to a shared library with the kernels
                        implementations.
  -c PATH_TO_CLDNN_CONFIG, --path_to_cldnn_config PATH_TO_CLDNN_CONFIG
                        Optional. Required for GPU custom kernels. Absolute
                        path to an .xml file with the kernels description.
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
                        Optional. Number of streams to use for inference on the CPU/GPU/MYRIAD
                        (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).
                        Default value is determined automatically for a device.
                        Please note that although the automatic selection usually provides a reasonable performance,
                        it still may be non-optimal for some cases, especially for very small networks.
                        Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency
                        estimations the number of streams should be set to 1.
  -enforcebf16 [{true,false}], --enforce_bfloat16 [{true,false}]
                        Optional. By default floating point operations execution in bfloat16 precision are enforced if supported by platform.
                        'true'  - enable  bfloat16 regardless of platform support
                        'false' - disable bfloat16 regardless of platform support.
  -nthreads NUMBER_THREADS, --number_threads NUMBER_THREADS
                        Number of threads to use for inference on the CPU
                        (including HETERO  and MULTI cases).
  -pin {YES,NO,NUMA}, --infer_threads_pinning {YES,NO,NUMA}
                        Optional. Enable threads->cores ('YES' is default
                        value), threads->(NUMA)nodes ('NUMA') or completely
                        disable ('NO')CPU threads pinning for CPU-involved
                        inference.
  --exec_graph_path EXEC_GRAPH_PATH
                        Optional. Path to a file where to store executable
                        graph information serialized.
  -pc [PERF_COUNTS], --perf_counts [PERF_COUNTS]
                        Optional. Report performance counters.
  -dump_config DUMP_CONFIG
                        Optional. Path to JSON file to dump IE parameters,
                        which were set by application.
  -load_config LOAD_CONFIG
                        Optional. Path to JSON file to load custom IE
                        parameters. Please note, command line parameters have
                        higher priority then parameters from configuration
                        file.
  -cdir CACHE_DIR, --cache_dir CACHE_DIR
                        Optional. Enable model caching to specified directory
  -lfile [LOAD_FROM_FILE], --load_from_file [LOAD_FROM_FILE]
                        Optional. Loads model from file directly without
                        read_network.
```

Running the application with the empty list of options yields the usage message given above and an error message.

Application supports topologies with one or more inputs. If a topology is not data sensitive, you can skip the input parameter. In this case, inputs are filled with random values.
If a model has only image input(s), please a provide folder with images or a path to an image as input.
If a model has some specific input(s) (not images), please prepare a binary file(s), which is filled with data of appropriate precision and provide a path to them as input.
If a model has mixed input types, input folder should contain all required files. Image inputs are filled with image files one by one. Binary inputs are filled with binary inputs one by one.

To run the tool, you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

For example, to do inference of an image using a trained network with multiple outputs on CPU, run the following command:

```
python3 benchmark_app.py -i <path_to_image>/inputImage.bmp -m <path_to_model>/multiple-output.xml -d CPU
```

## Demo Output

The application outputs number of executed iterations, total duration of execution, latency and throughput.
Additionally, if you set the `-pc` parameter, the application outputs performance counters.
If you set `-exec_graph_path`, the application reports executable graph information serialized.

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

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
