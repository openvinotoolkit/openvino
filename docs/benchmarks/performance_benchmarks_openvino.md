# Intel® Distribution of OpenVINO™ toolkit Benchmark Results {#openvino_docs_performance_benchmarks_openvino}

@sphinxdirective
.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_performance_benchmarks_faq
   Download Performance Data Spreadsheet in MS Excel* Format <https://docs.openvino.ai/downloads/benchmark_files/OV-2022.1-Download-Excel.xlsx>
   openvino_docs_performance_int8_vs_fp32


@endsphinxdirective

This benchmark setup includes a single machine on which both the benchmark application and the OpenVINO™ installation reside.

The benchmark application loads the OpenVINO™ Runtime and executes inferences on the specified hardware (CPU, GPU or VPU). The benchmark application measures the time spent on actual inferencing (excluding any pre or post processing) and then reports on the inferences per second (or Frames Per Second). For more information on the benchmark application, please also refer to the entry 5 of the [FAQ section](performance_benchmarks_faq.md).

Measuring inference performance involves many variables and is extremely use-case and application dependent. We use the below four parameters for measurements, which are key elements to consider for a successful deep learning inference application:

- **Throughput** - Measures the number of inferences delivered within a latency threshold. (for example, number of Frames Per Second - FPS). When deploying a system with deep learning inference, select the throughput that delivers the best trade-off between latency and power for the price and performance that meets your requirements.
- **Value** - While throughput is important, what is more critical in edge AI deployments is the performance efficiency or performance-per-cost. Application performance in throughput per dollar of system cost is the best measure of value.
- **Efficiency** - System power is a key consideration from the edge to the data center. When selecting deep learning solutions, power efficiency (throughput/watt) is a critical factor to consider. Intel designs provide excellent power efficiency for running deep learning workloads.
- **Latency** - This measures the synchronous execution of inference requests and is reported in milliseconds. Each inference request (for example: preprocess, infer, postprocess) is allowed to complete before the next is started. This performance metric is relevant in usage scenarios where a single image input needs to be acted upon as soon as possible. An example would be the healthcare sector where medical personnel only request analysis of a single ultra sound scanning image or in real-time or near real-time applications for example an industrial robot's response to actions in its environment or obstacle avoidance for autonomous vehicles.

## bert-base-cased [124]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/bert-base-cased124.csv"></div>

@endsphinxdirective


## bert-large-uncased-whole-word-masking-squad-int8-0001 [384]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/bert-large-uncased-whole-word-masking-squad-int8-0001-384.csv"></div>

@endsphinxdirective

## deeplabv3-TF [513x513]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/deeplabv3-TF-513x513.csv"></div>

@endsphinxdirective

## densenet-121-TF [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/densenet-121-TF-224x224.csv"></div>

@endsphinxdirective

## efficientdet-d0 [512x512]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/efficientdet-d0-512x512.csv"></div>

@endsphinxdirective

## faster-rcnn-resnet50-coco-TF [600x1024]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/faster_rcnn_resnet50_coco-TF-600x1024.csv"></div>

@endsphinxdirective

## inception-v4-TF [299x299]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/inception-v4-TF-299x299.csv"></div>

@endsphinxdirective

## mobilenet-ssd-CF [300x300]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/mobilenet-ssd-CF-300x300.csv"></div>

@endsphinxdirective

## mobilenet-v2-pytorch [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/mobilenet-v2-pytorch-224x224.csv"></div>

@endsphinxdirective

## resnet-18-pytorch [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/resnet-18-pytorch-224x224.csv"></div>

@endsphinxdirective


## resnet_50_TF [224x224]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/resnet-50-TF-224x224.csv"></div>

@endsphinxdirective

## ssd-resnet34-1200-onnx [1200x1200]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/ssd-resnet34-1200-onnx-1200x1200.csv"></div>

@endsphinxdirective

## unet-camvid-onnx-0001 [368x480]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/unet-camvid-onnx-0001-368x480.csv"></div>

@endsphinxdirective

## yolo-v3-tiny-tf [416x416]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/yolo-v3-tiny-tf-416x416.csv"></div>

@endsphinxdirective

## yolo_v4-tf [608x608]

@sphinxdirective
.. raw:: html

    <div class="chart-block" data-loadcsv="csv/yolo_v4-tf-608x608.csv"></div>

@endsphinxdirective

## Platform Configurations

Intel® Distribution of OpenVINO™ toolkit performance benchmark numbers are based on release 2022.1.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Learn more at intel.com, or from the OEM or retailer. Performance results are based on testing as of March 17, 2022 and may not reflect all publicly available updates. See configuration disclosure for details. No product can be absolutely secure.

Performance varies by use, configuration and other factors. Learn more at [www.intel.com/PerformanceIndex](https://www.intel.com/PerformanceIndex).

Your costs and results may vary.

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.

Intel optimizations, for Intel compilers or other products, may not optimize to the same degree for non-Intel products.

Testing by Intel done on: see test date for each HW platform below.

**CPU Inference Engines**

| Configuration                           | Intel® Xeon® E-2124G               | Intel® Xeon® W1290P                |
| -------------------------------         | ----------------------             | ---------------------------        |
| Motherboard                             | ASUS* WS C246 PRO                  | ASUS* WS W480-ACE                  |
| CPU                                     | Intel® Xeon® E-2124G CPU @ 3.40GHz | Intel® Xeon® W-1290P CPU @ 3.70GHz |
| Hyper Threading                         | OFF                                | ON                                 |
| Turbo Setting                           | ON                                 | ON                                 |
| Memory                                  | 2 x 16 GB DDR4 2666MHz             | 4 x 16 GB DDR4 @ 2666MHz           |
| Operating System                        | Ubuntu* 20.04.3 LTS                | Ubuntu* 20.04.3 LTS                |
| Kernel Version                          | 5.4.0-42-generic                   | 5.4.0-42-generic                   |
| BIOS Vendor                             | American Megatrends Inc.*          | American Megatrends Inc.           |
| BIOS Version                            | 1901                               | 2301                               |
| BIOS Release                            | September 24, 2021                 | July 8, 2021                       |
| BIOS Settings                           | Select optimized default settings, <br>save & exit | Select optimized default settings, <br>save & exit |
| Batch size                              | 1                                  | 1                                  |
| Precision                               | INT8                               | INT8                               |
| Number of concurrent inference requests | 4                                  | 5                                  |
| Test Date                               | March 17, 2022                     | March 17, 2022                      |
| Rated maximum TDP/socket in Watt        | [71](https://ark.intel.com/content/www/us/en/ark/products/134854/intel-xeon-e-2124g-processor-8m-cache-up-to-4-50-ghz.html#tab-blade-1-0-1)  | [125](https://ark.intel.com/content/www/us/en/ark/products/199336/intel-xeon-w-1290p-processor-20m-cache-3-70-ghz.html) |
| CPU Price/socket on Feb 14, 2022, USD<br>Prices may vary  | [213](https://ark.intel.com/content/www/us/en/ark/products/134854/intel-xeon-e-2124g-processor-8m-cache-up-to-4-50-ghz.html) | [539](https://ark.intel.com/content/www/us/en/ark/products/199336/intel-xeon-w-1290p-processor-20m-cache-3-70-ghz.html) |

**CPU Inference Engines (continue)**

| Configuration                           | Intel® Xeon® Silver 4216R               | Intel® Xeon® Silver 4316               |
| -------------------------------         | ----------------------                  | ---------------------------            |
| Motherboard                             | Intel® Server Board S2600STB            | Intel Corporation / WilsonCity         |
| CPU                                     | Intel® Xeon® Silver 4216R CPU @ 2.20GHz | Intel® Xeon® Silver 4316 CPU @ 2.30GHz |
| Hyper Threading                         | ON                                      | ON                                     |
| Turbo Setting                           | ON                                      | ON                                     |
| Memory                                  | 12 x 32 GB DDR4 2666MHz                 | 16 x 32 GB DDR4 @ 2666MHz              |
| Operating System                        | Ubuntu* 20.04.3 LTS                     | Ubuntu* 20.04.3 LTS                    |
| Kernel Version                          | 5.3.0-24-generic                        | 5.4.0-100-generic                      |
| BIOS Vendor                             | Intel Corporation                       | Intel Corporation                      |
| BIOS Version                            | SE5C620.86B.02.01.<br>0013.121520200651 | WLYDCRB1.SYS.0021.<br>P41.2109200451   |
| BIOS Release                            | December 15, 2020                       | September 20, 2021                     |
| BIOS Settings                           | Select optimized default settings, <br>change power policy <br>to "performance", <br>save & exit | Select optimized default settings, <br>save & exit |
| Batch size                              | 1                                       | 1                                      |
| Precision                               | INT8                                    | INT8                                   |
| Number of concurrent inference requests | 32                                      | 10                                     |
| Test Date                               | March 17, 2022                          | March 17, 2022                          |
| Rated maximum TDP/socket in Watt        | [125](https://ark.intel.com/content/www/us/en/ark/products/193394/intel-xeon-silver-4216-processor-22m-cache-2-10-ghz.html#tab-blade-1-0-1) | [150](https://ark.intel.com/content/www/us/en/ark/products/215270/intel-xeon-silver-4316-processor-30m-cache-2-30-ghz.html)|
| CPU Price/socket on June 21, 2021, USD<br>Prices may vary  | [1,002](https://ark.intel.com/content/www/us/en/ark/products/193394/intel-xeon-silver-4216-processor-22m-cache-2-10-ghz.html) | [1083](https://ark.intel.com/content/www/us/en/ark/products/215270/intel-xeon-silver-4316-processor-30m-cache-2-30-ghz.html)|

**CPU Inference Engines (continue)**

| Configuration                           | Intel® Xeon® Gold 5218T                 | Intel® Xeon® Platinum 8270               | Intel® Xeon® Platinum 8380               |
| -------------------------------         | ----------------------------            | ----------------------------             | -----------------------------------------|
| Motherboard                             | Intel® Server Board S2600STB            | Intel® Server Board S2600STB             | Intel Corporation / WilsonCity           |
| CPU                                     | Intel® Xeon® Gold 5218T CPU @ 2.10GHz   | Intel® Xeon® Platinum 8270 CPU @ 2.70GHz | Intel® Xeon® Platinum 8380 CPU @ 2.30GHz |
| Hyper Threading                         | ON                                      | ON                                       | ON                                       |
| Turbo Setting                           | ON                                      | ON                                       | ON                                       |
| Memory                                  | 12 x 32 GB DDR4 2666MHz                 | 12 x 32 GB DDR4 2933MHz                  | 16 x 16 GB DDR4 3200MHz                  |
| Operating System                        | Ubuntu* 20.04.3 LTS                     | Ubuntu* 20.04.3 LTS                      | Ubuntu* 20.04.1 LTS                        |
| Kernel Version                          | 5.3.0-24-generic                        | 5.3.0-24-generic                         | 5.4.0-64-generic                         |
| BIOS Vendor                             | Intel Corporation                       | Intel Corporation                        | Intel Corporation                        |
| BIOS Version                            | SE5C620.86B.02.01.<br>0013.121520200651 | SE5C620.86B.02.01.<br>0013.121520200651  | WLYDCRB1.SYS.0020.<br>P86.2103050636     |
| BIOS Release                            | December 15, 2020                       | December 15, 2020                        | March 5, 2021                            |
| BIOS Settings                           | Select optimized default settings, <br>change power policy to "performance", <br>save & exit | Select optimized default settings, <br>change power policy to "performance", <br>save & exit | Select optimized default settings, <br>change power policy to "performance", <br>save & exit |
| Batch size                              | 1                                       | 1                                        | 1                                        |
| Precision                               | INT8                                    | INT8                                     | INT8                                     |
| Number of concurrent inference requests | 32                                      | 52                                       | 80                                       |
| Test Date                               | March 17, 2022                          | March 17, 2022                            | March 17, 2022                            |
| Rated maximum TDP/socket in Watt        | [105](https://ark.intel.com/content/www/us/en/ark/products/193953/intel-xeon-gold-5218t-processor-22m-cache-2-10-ghz.html#tab-blade-1-0-1)           | [205](https://ark.intel.com/content/www/us/en/ark/products/192482/intel-xeon-platinum-8270-processor-35-75m-cache-2-70-ghz.html#tab-blade-1-0-1) | [270](https://mark.intel.com/content/www/us/en/secure/mark/products/212287/intel-xeon-platinum-8380-processor-60m-cache-2-30-ghz.html#tab-blade-1-0-1) |
| CPU Price/socket on Feb 14, 2022, USD<br>Prices may vary  | [1,349](https://ark.intel.com/content/www/us/en/ark/products/193953/intel-xeon-gold-5218t-processor-22m-cache-2-10-ghz.html) | [7,405](https://ark.intel.com/content/www/us/en/ark/products/192482/intel-xeon-platinum-8270-processor-35-75m-cache-2-70-ghz.html) | [8,099](https://mark.intel.com/content/www/us/en/secure/mark/products/212287/intel-xeon-platinum-8380-processor-60m-cache-2-30-ghz.html#tab-blade-1-0-0) |


**CPU Inference Engines (continue)**

| Configuration        | Intel® Core™ i9-10920X               | Intel® Core™ i9-10900TE                 | Intel® Core™ i9-12900                                          |
| -------------------- | -------------------------------------| -----------------------                 | -------------------------------------------------------------- |
| Motherboard          | ASUS* PRIME X299-A II                | B595                                    | Intel Corporation<br>internal/Reference<br>Validation Platform |
| CPU                  | Intel® Core™ i9-10920X CPU @ 3.50GHz | Intel® Core™ i9-10900TE CPU @ 1.80GHz   | 12th Gen Intel® Core™ i9-12900                                 |
| Hyper Threading      | ON                                   | ON                                      | OFF                                                            |
| Turbo Setting        | ON                                   | ON                                      | -                                                              |
| Memory               | 4 x 16 GB DDR4 2666MHz               | 2 x 8 GB DDR4 @ 2400 MHz                | 4 x 8 GB DDR4 4800MHz                                          |
| Operating System     | Ubuntu 20.04.3 LTS                   | Ubuntu 20.04.3 LTS                      | Microsoft Windows 10 Pro                                       |
| Kernel Version       | 5.4.0-42-generic                     | 5.4.0-42-generic                        | 10.0.19043 N/A Build 19043                                     |
| BIOS Vendor          | American Megatrends Inc.*            | American Megatrends Inc.*               | Intel Corporation                                              |
| BIOS Version         | 1004                                 | Z667AR10.BIN                            | ADLSFWI1.R00.2303.<br>B00.2107210432                           |
| BIOS Release         | March 19, 2021                       | July 15, 2020                           | July 21, 2021                                                  |
| BIOS Settings        | Default Settings                     | Default Settings                        | Default Settings                                               |
| Batch size           | 1                                    | 1                                       | 1                                                              |
| Precision            | INT8                                 | INT8                                    | INT8                                                           |
| Number of concurrent inference requests | 24                | 5                                       | 4                                                              |
| Test Date            | March 17, 2022                       | March 17, 2022                          | March 17, 2022                                                 |
| Rated maximum TDP/socket in Watt                            | [165](https://ark.intel.com/content/www/us/en/ark/products/198012/intel-core-i9-10920x-x-series-processor-19-25m-cache-3-50-ghz.html) | [35](https://ark.intel.com/content/www/us/en/ark/products/203901/intel-core-i910900te-processor-20m-cache-up-to-4-60-ghz.html)  | [65](https://ark.intel.com/content/www/us/en/ark/products/134597/intel-core-i912900-processor-30m-cache-up-to-5-10-ghz.html) |
| CPU Price/socket on Feb 14, 2022, USD<br>Prices may vary    | [700](https://ark.intel.com/content/www/us/en/ark/products/198012/intel-core-i9-10920x-x-series-processor-19-25m-cache-3-50-ghz.html) | [444](https://ark.intel.com/content/www/us/en/ark/products/203901/intel-core-i910900te-processor-20m-cache-up-to-4-60-ghz.html) | [519](https://ark.intel.com/content/www/us/en/ark/products/134597/intel-core-i912900-processor-30m-cache-up-to-5-10-ghz.html)|

**CPU Inference Engines (continue)**
| Configuration        | Intel® Core™ i7-8700T               | Intel® Core™ i7-1185G7                                         |
| -------------------- | ----------------------------------- | --------------------------------                               |
| Motherboard          | GIGABYTE* Z370M DS3H-CF             | Intel Corporation<br>internal/Reference<br>Validation Platform |
| CPU                  | Intel® Core™ i7-8700T CPU @ 2.40GHz | Intel® Core™ i7-1185G7 @ 3.00GHz                               |
| Hyper Threading      | ON                                  | ON                                                             |
| Turbo Setting        | ON                                  | ON                                                             |
| Memory               | 4 x 16 GB DDR4 2400MHz              | 2 x 8 GB DDR4 3200MHz                                          |
| Operating System     | Ubuntu 20.04.3 LTS                  | Ubuntu 20.04.3 LTS                                             |
| Kernel Version       | 5.4.0-42-generic                    | 5.8.0-050800-generic                                           |
| BIOS Vendor          | American Megatrends Inc.*           | Intel Corporation                                              |
| BIOS Version         | F14c                                | TGLSFWI1.R00.4391.<br>A00.2109201819                           |
| BIOS Release         | March 23, 2021                      | September 20, 2021                                             |
| BIOS Settings        | Select optimized default settings, <br>set OS type to "other", <br>save & exit | Default Settings    |
| Batch size           | 1                                   | 1                                                              |
| Precision            | INT8                                | INT8                                                           |
| Number of concurrent inference requests | 4                | 4                                                              |
| Test Date            | March 17, 2022                      | March 17, 2022                                                 |
| Rated maximum TDP/socket in Watt                           | [35](https://ark.intel.com/content/www/us/en/ark/products/129948/intel-core-i7-8700t-processor-12m-cache-up-to-4-00-ghz.html#tab-blade-1-0-1) | [28](https://ark.intel.com/content/www/us/en/ark/products/208664/intel-core-i7-1185g7-processor-12m-cache-up-to-4-80-ghz-with-ipu.html) |
| CPU Price/socket on Feb 14, 2022, USD<br>Prices may vary   | [303](https://ark.intel.com/content/www/us/en/ark/products/129948/intel-core-i7-8700t-processor-12m-cache-up-to-4-00-ghz.html) | [426](https://ark.intel.com/content/www/us/en/ark/products/208664/intel-core-i7-1185g7-processor-12m-cache-up-to-4-80-ghz-with-ipu.html)               |

**CPU Inference Engines (continue)**

| Configuration        | Intel® Core™ i3-8100               | Intel® Core™ i5-8500               | Intel® Core™ i5-10500TE               |
| -------------------- |----------------------------------- | ---------------------------------- | -----------------------------------   |
| Motherboard          | GIGABYTE* Z390 UD                  | ASUS* PRIME Z370-A                 | GIGABYTE* Z490 AORUS PRO AX           |
| CPU                  | Intel® Core™ i3-8100 CPU @ 3.60GHz | Intel® Core™ i5-8500 CPU @ 3.00GHz | Intel® Core™ i5-10500TE CPU @ 2.30GHz |
| Hyper Threading      | OFF                                | OFF                                | ON                                    |
| Turbo Setting        | OFF                                | ON                                 | ON                                    |
| Memory               | 4 x 8 GB DDR4 2400MHz              | 2 x 16 GB DDR4 2666MHz             | 2 x 16 GB DDR4 @ 2666MHz              |
| Operating System     | Ubuntu* 20.04.3 LTS                | Ubuntu* 20.04.3 LTS                | Ubuntu* 20.04.3 LTS                   |
| Kernel Version       | 5.3.0-24-generic                   | 5.4.0-42-generic                   | 5.4.0-42-generic                      |
| BIOS Vendor          | American Megatrends Inc.*          | American Megatrends Inc.*          | American Megatrends Inc.*             |
| BIOS Version         | F8                                 | 3004                               | F21                                   |
| BIOS Release         | May 24, 2019                       | July 12, 2021                      | November 23, 2021                     |
| BIOS Settings        | Select optimized default settings, <br> set OS type to "other", <br>save & exit | Select optimized default settings, <br>save & exit | Select optimized default settings, <br>set OS type to "other", <br>save & exit |
| Batch size           | 1                                  | 1                                  | 1                                     |
| Precision            | INT8                               | INT8                               | INT8                                  |
| Number of concurrent inference requests | 4               | 3                                  | 4                                     |
| Test Date            | March 17, 2022                     | March 17, 2022                     | March 17, 2022                        |
| Rated maximum TDP/socket in Watt                          | [65](https://ark.intel.com/content/www/us/en/ark/products/126688/intel-core-i3-8100-processor-6m-cache-3-60-ghz.html#tab-blade-1-0-1)| [65](https://ark.intel.com/content/www/us/en/ark/products/129939/intel-core-i5-8500-processor-9m-cache-up-to-4-10-ghz.html#tab-blade-1-0-1)| [35](https://ark.intel.com/content/www/us/en/ark/products/203891/intel-core-i5-10500te-processor-12m-cache-up-to-3-70-ghz.html)  |
| CPU Price/socket on Feb 14, 2022, USD<br>Prices may vary  | [117](https://ark.intel.com/content/www/us/en/ark/products/126688/intel-core-i3-8100-processor-6m-cache-3-60-ghz.html) | [192](https://ark.intel.com/content/www/us/en/ark/products/129939/intel-core-i5-8500-processor-9m-cache-up-to-4-10-ghz.html)               | [195](https://ark.intel.com/content/www/us/en/ark/products/203891/intel-core-i5-10500te-processor-12m-cache-up-to-3-70-ghz.html) |


**CPU Inference Engines (continue)**

| Configuration        | Intel Atom® x5-E3940                  | Intel Atom® x6425RE                               | Intel® Celeron® 6305E            |
| -------------------- | --------------------------------------|-------------------------------                    |----------------------------------|
| Motherboard          | Intel Corporation<br>internal/Reference<br>Validation Platform | Intel Corporation<br>internal/Reference<br>Validation Platform | Intel Corporation<br>internal/Reference<br>Validation Platform      |
| CPU                  | Intel Atom® Processor E3940 @ 1.60GHz | Intel Atom® x6425RE<br>Processor @ 1.90GHz        | Intel® Celeron®<br>6305E @ 1.80GHz  |
| Hyper Threading      | OFF                                   | OFF                                               | OFF |
| Turbo Setting        | ON                                    | ON                                                | ON |
| Memory               | 1 x 8 GB DDR3 1600MHz                 | 2 x 4GB DDR4 3200MHz                              | 2 x 8 GB DDR4 3200MHz |
| Operating System     | Ubuntu* 20.04.3 LTS                   | Ubuntu* 20.04.3 LTS                               | Ubuntu 20.04.3 LTS               |
| Kernel Version       | 5.4.0-42-generic                      | 5.13.0-27-generic                                 | 5.13.0-1008-intel |
| BIOS Vendor          | American Megatrends Inc.*             | Intel Corporation                                 | Intel Corporation |
| BIOS Version         | 5.12                                  | EHLSFWI1.R00.3273.<br>A01.2106300759              | TGLIFUI1.R00.4064.A02.2102260133 |
| BIOS Release         | September 6, 2017                     | June 30, 2021                                     | February 26, 2021 |
| BIOS Settings        | Default settings                      | Default settings                                  | Default settings |
| Batch size           | 1                                     | 1                                                 | 1 |
| Precision            | INT8                                  | INT8                                              | INT8 |
| Number of concurrent inference requests | 4                  | 4                                                 | 4|
| Test Date            | March 17, 2022                        | March 17, 2022                                    | March 17, 2022 |
| Rated maximum TDP/socket in Watt | [9.5](https://ark.intel.com/content/www/us/en/ark/products/96485/intel-atom-x5-e3940-processor-2m-cache-up-to-1-80-ghz.html)  | [12](https://mark.intel.com/content/www/us/en/secure/mark/products/207907/intel-atom-x6425e-processor-1-5m-cache-up-to-3-00-ghz.html#tab-blade-1-0-1) | [15](https://ark.intel.com/content/www/us/en/ark/products/208072/intel-celeron-6305e-processor-4m-cache-1-80-ghz.html)|
| CPU Price/socket on Feb 14, 2022, USD<br>Prices may vary  | [34](https://ark.intel.com/content/www/us/en/ark/products/96485/intel-atom-x5-e3940-processor-2m-cache-up-to-1-80-ghz.html) | [59](https://ark.intel.com/content/www/us/en/ark/products/207899/intel-atom-x6425re-processor-1-5m-cache-1-90-ghz.html) |[107](https://ark.intel.com/content/www/us/en/ark/products/208072/intel-celeron-6305e-processor-4m-cache-1-80-ghz.html) |

**Accelerator Inference Engines**

| Configuration                           | Intel® Neural Compute Stick 2         | Intel® Vision Accelerator Design<br>with Intel® Movidius™ VPUs (Mustang-V100-MX8) |
| --------------------------------------- | ------------------------------------- | ------------------------------------- |
| VPU                                     | 1 X Intel® Movidius™ Myriad™ X MA2485 | 8 X Intel® Movidius™ Myriad™ X MA2485 |
| Connection                              | USB 2.0/3.0                           | PCIe X4                               |
| Batch size                              | 1                                     | 1                                     |
| Precision                               | FP16                                  | FP16                                  |
| Number of concurrent inference requests | 4                                     | 32                                    |
| Rated maximum TDP/socket in Watt        | 2.5                                   | [30](https://www.mouser.com/ProductDetail/IEI/MUSTANG-V100-MX8-R10?qs=u16ybLDytRaZtiUUvsd36w%3D%3D)          |
| CPU Price/socket on Feb 14, 2022, USD<br>Prices may vary | [69](https://ark.intel.com/content/www/us/en/ark/products/140109/intel-neural-compute-stick-2.html) | [492](https://www.mouser.com/ProductDetail/IEI/MUSTANG-V100-MX8-R10?qs=u16ybLDytRaZtiUUvsd36w%3D%3D)  |
| Host Computer                           | Intel® Core™ i7                       | Intel® Core™ i5                       |
| Motherboard                             | ASUS* Z370-A II                       | Uzelinfo* / US-E1300                  |
| CPU                                     | Intel® Core™ i7-8700 CPU @ 3.20GHz    | Intel® Core™ i5-6600 CPU @ 3.30GHz    |
| Hyper Threading                         | ON                                    | OFF                                   |
| Turbo Setting                           | ON                                    | ON                                    |
| Memory                                  | 4 x 16 GB DDR4 2666MHz                | 2 x 16 GB DDR4 2400MHz                |
| Operating System                        | Ubuntu* 20.04.3 LTS                   | Ubuntu* 20.04.3 LTS                   |
| Kernel Version                          | 5.0.0-23-generic                      | 5.0.0-23-generic                      |
| BIOS Vendor                             | American Megatrends Inc.*             | American Megatrends Inc.*             |
| BIOS Version                            | 411                                   | 5.12                                  |
| BIOS Release                            | September 21, 2018                    | September 21, 2018                    |
| Test Date                               | March 17, 2022                        | March 17, 2022                        |

For more detailed configuration descriptions, see [Configuration Details](https://docs.openvino.ai/resources/benchmark_files/system_configurations_2022.1.html).