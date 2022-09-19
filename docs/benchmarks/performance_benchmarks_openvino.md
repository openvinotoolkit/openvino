# Intel® Distribution of OpenVINO™ toolkit Benchmark Results {#openvino_docs_performance_benchmarks_openvino}

@sphinxdirective
.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_performance_benchmarks_faq
   openvino_docs_performance_int8_vs_fp32
   Performance Data Spreadsheet (download xlsx) <https://docs.openvino.ai/2022.2/_static/benchmarks_files/OV-2022.2-Performance-Data.xlsx>

@endsphinxdirective


## Benchmark Setup Information

This benchmark setup includes a single machine on which both the benchmark application and the OpenVINO™ installation reside. The presented performance benchmark numbers are based on release 2022.2 of the Intel® Distribution of OpenVINO™ toolkit.

The benchmark application loads the OpenVINO™ Runtime and executes inference on the specified hardware (CPU, GPU or VPU). It measures the time spent on actual inferencing (excluding any pre or post processing) and then reports on the inferences per second (or Frames Per Second - FPS). For additional information on the benchmark application, refer to the entry 5 in the [FAQ section](performance_benchmarks_faq.md).

Measuring inference performance involves many variables and is extremely use-case and application dependent. Below are four parameters used for measurements, which are key elements to consider for a successful deep learning inference application:

- **Throughput** - Measures the number of inferences delivered within a latency threshold (for example, number of FPS). When deploying a system with deep learning inference, select the throughput that delivers the best trade-off between latency and power for the price and performance that meets your requirements.
- **Value** - While throughput is important, what is more critical in edge AI deployments is the performance efficiency or performance-per-cost. Application performance in throughput per dollar of system cost is the best measure of value.
- **Efficiency** - System power is a key consideration from the edge to the data center. When selecting deep learning solutions, power efficiency (throughput/watt) is a critical factor to consider. Intel designs provide excellent power efficiency for running deep learning workloads.
- **Latency** - This parameter measures the synchronous execution of inference requests and is reported in milliseconds. Each inference request (i.e., preprocess, infer, postprocess) is allowed to complete before the next one is started. This performance metric is relevant in usage scenarios where a single image input needs to be acted upon as soon as possible. An example of that kind of a scenario would be real-time or near real-time applications, i.e., the response of an industrial robot to its environment or obstacle avoidance for autonomous vehicles.

For a listing of all platforms and configurations used for testing, refer to the following:
@sphinxdirective
* :download:`HW platforms (pdf) <_static/benchmarks_files/platform_list_22.2.pdf>`
* :download:`Configuration Details (xlsx) <_static/benchmarks_files/OV-2022.2-system-info-detailed.xlsx>`

@endsphinxdirective


## Benchmark Performance Results

Benchmark performance results below are based on testing as of September 20, 2022. They may not reflect all publicly available updates at the time of testing.
<!-- See configuration disclosure for details. No product can be absolutely secure. -->
Performance varies by use, configuration and other factors, which are elaborated further in [here](https://www.intel.com/PerformanceIndex). Used Intel optimizations (for Intel® compilers or other products) may not optimize to the same degree for non-Intel products.

### bert-base-cased_onnx [124]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/bert-base-cased_onnx.csv"></div>

@endsphinxdirective


### bert-large-uncased-whole-word-masking-squad-0001_onnx [384]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/bert-large-uncased-whole-word-masking-squad-0001_onnx.csv"></div>

@endsphinxdirective

### deeplabv3_tf [513x513]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/deeplabv3_tf.csv"></div>

@endsphinxdirective

### densenet-121_tf [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/densenet-121_tf.csv"></div>

@endsphinxdirective

### efficientdet-d0_tf [512x512]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/efficientdet-d0_tf.csv"></div>

@endsphinxdirective

### mask_rcnn_resnet50_atrous_coco_tf [600x1024]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/mask_rcnn_resnet50_atrous_coco_tf.csv"></div>

@endsphinxdirective

### ssd-resnet34-1200_onnx [1200x1200]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/ssd-resnet34-1200_onnx.csv"></div>

@endsphinxdirective

### resnet-50_tf [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/resnet-50_tf.csv"></div>

@endsphinxdirective

### resnet-50-pytorch_onnx [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/resnet-50-pytorch_onnx.csv"></div>

@endsphinxdirective


### yolo_v3_tiny_tf [416x416]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/yolo_v3_tiny_tf.csv"></div>

@endsphinxdirective

### yolo_v4_tf2 [608x608]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/yolo_v4_tf2.csv"></div>

@endsphinxdirective

### googlenet-v4_tf [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/googlenet-v4_tf.csv"></div>

@endsphinxdirective

### ssd_mobilenet_v1_coco_tf [300x300]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/ssd_mobilenet_v1_coco_tf.csv"></div>

@endsphinxdirective

### ssd_mobilenet_v2_coco_tf [300x300]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/ssd_mobilenet_v2_coco_tf.csv"></div>

@endsphinxdirective

### unet-camvid-onnx-0001_onnx [368x480]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/unet-camvid-onnx-0001_onnx.csv"></div>

@endsphinxdirective




## Disclaimers
Intel® Distribution of OpenVINO™ toolkit performance benchmark numbers are based on release 2022.2.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Learn more at intel.com, or from the OEM or retailer. Performance results are based on testing as of September 20, 2022 and may not reflect all publicly available updates. See configuration disclosure for details. No product can be absolutely secure.

Performance varies by use, configuration and other factors. Learn more at [www.intel.com/PerformanceIndex](https://www.intel.com/PerformanceIndex).

Your costs and results may vary.

Intel optimizations, for Intel compilers or other products, may not optimize to the same degree for non-Intel products.

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.
