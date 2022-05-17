# Intel® Distribution of OpenVINO™ toolkit Benchmark Results {#openvino_docs_performance_benchmarks_openvino}

@sphinxdirective
.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_performance_benchmarks_faq
   Download Performance Data Spreadsheet in MS Excel Format <https://docs.openvino.ai/downloads/benchmark_files/OV-2022.1-Download-Excel.xlsx>
   openvino_docs_performance_int8_vs_fp32


@endsphinxdirective

Features and benefits of Intel® technologies depend on system configuration and may require enabled hardware, software or service activation. More information on this subject may be obtained from the original equipment manufacturer (OEM), official [Intel® web page](https://www.intel.com) or retailer.

## Platform Configurations

@sphinxdirective

:download:`A full list of used for testing HW platforms (along with their configuration)<../../../docs/benchmarks/files/Platform_list.pdf>`

@endsphinxdirective

For more detailed configuration descriptions, see the [Configuration Details](https://docs.openvino.ai/resources/benchmark_files/system_configurations_2022.1.html) document.

## Benchmark Setup Information

This benchmark setup includes a single machine on which both the benchmark application and the OpenVINO™ installation reside. The presented performance benchmark numbers are based on realease 2022.1 of Intel® Distribution of OpenVINO™ toolkit.

The benchmark application loads the OpenVINO™ Runtime and executes inferences on the specified hardware (CPU, GPU or VPU). It measures the time spent on actual inferencing (excluding any pre or post processing) and then reports on the inferences per second (or Frames Per Second - FPS). For additional information on the benchmark application, refer to the entry 5 in the [FAQ section](performance_benchmarks_faq.md).

Measuring inference performance involves many variables and is extremely use case and application dependent. Below four parameters for measurements our used for measurrment. They are key elements to consider for a successful deep learning inference application:

- **Throughput** - Measures the number of inferences delivered within a latency threshold (for example, number of FPS). When deploying a system with deep learning inference, select the throughput that delivers the best trade-off between latency and power for the price and performance that meets your requirements.
- **Value** - While throughput is important, what is more critical in edge AI deployments is the performance efficiency or performance-per-cost. Application performance in throughput per dollar of system cost is the best measure of value.
- **Efficiency** - System power is a key consideration from the edge to the data center. When selecting deep learning solutions, power efficiency (throughput/watt) is a critical factor to consider. Intel designs provide excellent power efficiency for running deep learning workloads.
- **Latency** - This parameter measures the synchronous execution of inference requests and is reported in milliseconds. Each inference request (i.e., preprocess, infer, postprocess) is allowed to complete before the next is started. This performance metric is relevant in usage scenarios where a single image input needs to be acted upon as soon as possible. An example of that kind of scenario would be real-time or near real-time applications, i.e., an industrial robot's response to its environment or obstacle avoidance for autonomous vehicles.

## Benchmark Performance Results

Below benchmark performance results are based on testing as of March 17, 2022. They may not reflect all publicly available updates at the time of testing.
<!-- See configuration disclosure for details. No product can be absolutely secure. -->
Performance varies by use, configuration and other factors about which you can learn more [here](https://www.intel.com/PerformanceIndex). Used Intel optimizations (for Intel® compilers or other products) may not optimize to the same degree for non-Intel products.

### bert-base-cased [124]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/bert-base-cased124.csv"></div>

@endsphinxdirective


### bert-large-uncased-whole-word-masking-squad-int8-0001 [384]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/bert-large-uncased-whole-word-masking-squad-int8-0001-384.csv"></div>

@endsphinxdirective

### deeplabv3-TF [513x513]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/deeplabv3-TF-513x513.csv"></div>

@endsphinxdirective

### densenet-121-TF [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/densenet-121-TF-224x224.csv"></div>

@endsphinxdirective

### efficientdet-d0 [512x512]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/efficientdet-d0-512x512.csv"></div>

@endsphinxdirective

### faster-rcnn-resnet50-coco-TF [600x1024]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/faster_rcnn_resnet50_coco-TF-600x1024.csv"></div>

@endsphinxdirective

### inception-v4-TF [299x299]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/inception-v4-TF-299x299.csv"></div>

@endsphinxdirective

### mobilenet-ssd-CF [300x300]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/mobilenet-ssd-CF-300x300.csv"></div>

@endsphinxdirective

### mobilenet-v2-pytorch [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/mobilenet-v2-pytorch-224x224.csv"></div>

@endsphinxdirective

### resnet-18-pytorch [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/resnet-18-pytorch-224x224.csv"></div>

@endsphinxdirective


### resnet_50_TF [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/resnet-50-TF-224x224.csv"></div>

@endsphinxdirective

### ssd-resnet34-1200-onnx [1200x1200]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/ssd-resnet34-1200-onnx-1200x1200.csv"></div>

@endsphinxdirective

### unet-camvid-onnx-0001 [368x480]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/unet-camvid-onnx-0001-368x480.csv"></div>

@endsphinxdirective

### yolo-v3-tiny-tf [416x416]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/yolo-v3-tiny-tf-416x416.csv"></div>

@endsphinxdirective

### yolo_v4-tf [608x608]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/yolo_v4-tf-608x608.csv"></div>

@endsphinxdirective

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.
