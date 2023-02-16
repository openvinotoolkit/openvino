@sphinxdirective
:orphan:
@endsphinxdirective
# OpenVINO™ Model Server Benchmark Results {#openvino_docs_performance_benchmarks_ovms}

OpenVINO™ Model Server is an open-source, production-grade inference platform that exposes a set of models via a convenient inference API over gRPC or HTTP/REST. It employs the OpenVINO™ Runtime libraries from the Intel® Distribution of OpenVINO™ toolkit to extend workloads across Intel® hardware including CPU, GPU and others.

![OpenVINO™ Model Server](../img/performance_benchmarks_ovms_01.png)

## Measurement Methodology

OpenVINO™ Model Server is measured in multiple-client-single-server configuration using two hardware platforms connected by ethernet network. The network bandwidth depends on the platforms as well as models under investigation and it is set to not be a bottleneck for workload intensity. This connection is dedicated only to the performance measurements. The benchmark setup is consists of four main parts:

![OVMS Benchmark Setup Diagram](../img/performance_benchmarks_ovms_02.png)

* **OpenVINO™ Model Server** is launched as a docker container on the server platform and it listens (and answers on) requests from clients. OpenVINO™ Model Server is run on the same machine as the OpenVINO™ toolkit benchmark application in corresponding benchmarking. Models served by OpenVINO™ Model Server are located in a local file system mounted into the docker container. The OpenVINO™ Model Server instance communicates with other components via ports over a dedicated docker network.

* **Clients** are run in separated physical machine referred to as client platform. Clients are implemented in Python3 programming language based on TensorFlow* API and they work as parallel processes. Each client waits for a response from OpenVINO™ Model Server before it will send a new next request. The role played by the clients is also verification of responses.

* **Load balancer** works on the client platform in a docker container. HAProxy is used for this purpose. Its main role is counting of requests forwarded from clients to OpenVINO™ Model Server, estimating its latency, and sharing this information by Prometheus service. The reason of locating the load balancer on the client site is to simulate real life scenario that includes impact of physical network on reported metrics.

* **Execution Controller** is launched on the client platform. It is responsible for synchronization of the whole measurement process, downloading metrics from the load balancer, and presenting the final report of the execution.

## bert-small-uncased-whole-word-masking-squad-002 (INT8)
![](../_static/benchmarks_files/ovms/bert-small-uncased-whole-word-masking-squad-002-int8.png)
## bert-small-uncased-whole-word-masking-squad-002 (FP32)
![](../_static/benchmarks_files/ovms/bert-small-uncased-whole-word-masking-squad-002-fp32.png)
## densenet-121 (INT8)
![](../_static/benchmarks_files/ovms/densenet-121-int8.png)
## densenet-121 (FP32)
![](../_static/benchmarks_files/ovms/densenet-121-fp32.png)
## efficientdet-d0 (INT8)
![](../_static/benchmarks_files/ovms/efficientdet-d0-int8.png)
## efficientdet-d0 (FP32)
![](../_static/benchmarks_files/ovms/efficientdet-d0-fp32.png)
## inception-v4 (INT8)
![](../_static/benchmarks_files/ovms/inception-v4-int8.png)
## inception-v4 (FP32)
![](../_static/benchmarks_files/ovms/inception-v4-fp32.png)
## mobilenet-ssd (INT8)
![](../_static/benchmarks_files/ovms/mobilenet-ssd-int8.png)
## mobilenet-ssd (FP32)
![](../_static/benchmarks_files/ovms/mobilenet-ssd-fp32.png)
## mobilenet-v2 (INT8)
![](../_static/benchmarks_files/ovms/mobilenet-v2-int8.png)
## mobilenet-v2 (FP32)
![](../_static/benchmarks_files/ovms/mobilenet-v2-fp32.png)
## resnet-18 (INT8)
![](../_static/benchmarks_files/ovms/resnet-18-int8.png)
## resnet-18 (FP32)
![](../_static/benchmarks_files/ovms/resnet-18-fp32.png)
## resnet-50 (INT8)
![](../_static/benchmarks_files/ovms/resnet-50-int8.png)
## resnet-50 (FP32)
![](../_static/benchmarks_files/ovms/resnet-50-fp32.png)
## ssd-resnt34-1200 (INT8)
![](../_static/benchmarks_files/ovms/ssd-resnt34-1200-int8.png)
## ssd-resnt34-1200 (FP32)
![](../_static/benchmarks_files/ovms/ssd-resnt34-1200-fp32.png)
## unet-camvid-onnx-001 (INT8)
![](../_static/benchmarks_files/ovms/unet-camvid-onnx-001-int8.png)
## unet-camvid-onnx-001 (FP32)
![](../_static/benchmarks_files/ovms/unet-camvid-onnx-001-fp32.png)
## yolo-v3-tiny (INT8)
![](../_static/benchmarks_files/ovms/yolo-v3-tiny-int8.png)
## yolo-v3-tiny (FP32)
![](../_static/benchmarks_files/ovms/yolo-v3-tiny-fp32.png)
## yolo-v4 (INT8)
![](../_static/benchmarks_files/ovms/yolo-v4-int8.png)
## yolo-v4 (FP32)
![](../_static/benchmarks_files/ovms/yolo-v4-fp32.png)


## Platform Configurations

OpenVINO™ Model Server performance benchmark numbers are based on release 2022.2. Performance results are based on testing as of November 16, 2022 and may not reflect all publicly available updates.


@sphinxdirective
.. dropdown:: Platform with Intel® Xeon® Platinum 8260M

   .. table:: 
      :widths: 25 25 50

      +--------------------------+-------------------------------------------+----------------------------------------+
      |                          | Server Platform                           | Client Platform                        |
      +==========================+===========================================+========================================+
      | Motherboard              | Inspur YZMB-00882-104 NF5280M5            | Inspur YZMB-00882-104 NF5280M5         |
      +--------------------------+-------------------------------------------+----------------------------------------+
      | Memory                   | Samsung 16 x 16GB @ 2666 MT/s DDR4        | Kingston 16 x 16GB @ 2666 MT/s DDR4    |
      +--------------------------+-------------------------------------------+----------------------------------------+
      | CPU                      | Intel® Xeon® Platinum 8260M CPU @ 2.40GHz | Intel® Xeon® Gold 6238M CPU @ 2.10GHz  |
      +--------------------------+-------------------------------------------+----------------------------------------+
      | Selected CPU Flags       | Hyper Threading, Turbo Boost, DL Boost    | Hyper Threading, Turbo Boost, DL Boost |
      +--------------------------+-------------------------------------------+----------------------------------------+
      | CPU Thermal Design Power | 162W                                      | 150W                                   |
      +--------------------------+-------------------------------------------+----------------------------------------+
      | Operating System         | Ubuntu 20.04.4 LTS                        | Ubuntu 20.04.4 LTS                     |
      +--------------------------+-------------------------------------------+----------------------------------------+
      | Kernel Version           | 5.4.0-107-generic                         | 5.4.0-107-generic                      |
      +--------------------------+-------------------------------------------+----------------------------------------+
      | BIOS Vendor              | American Megatrends Inc.                  | AMI                                    |
      +--------------------------+-------------------------------------------+----------------------------------------+
      | BIOS Version & Release   | 4.1.16; date: 06/23/2020                  | 4.1.16; date: 06/23/2020               |
      +--------------------------+-------------------------------------------+----------------------------------------+
      | Docker Version           | 20.10.3                                   | 20.10.3                                |
      +--------------------------+-------------------------------------------+----------------------------------------+
      | Network Speed            | 40 Gb/s                                   | 40 Gb/s                                |
      +--------------------------+-------------------------------------------+----------------------------------------+

.. dropdown:: Platform with 6238M

      .. table:: 
         :widths: 25 25 50

         +--------------------------+-------------------------------------------+--------------------------------------------+
         |                          | Server Platform                           | Client Platform                            |
         +==========================+===========================================+============================================+
         | Motherboard              | Inspur YZMB-00882-104 NF5280M5            | Inspur YZMB-00882-104 NF5280M5             |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Memory                   | Kingston 16 x 16GB @ 2666 MT/s DDR4       | Samsung 16 x 16GB @ 2666 MT/s DDR4         |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | CPU                      | Intel® Xeon® Gold 6238M CPU @ 2.10GHz     | Intel® Xeon® Platinum 8260M CPU @ 2.40GHz  |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Selected CPU Flags       | Hyper Threading, Turbo Boost, DL Boost    | Hyper Threading, Turbo Boost, DL Boost     |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | CPU Thermal Design Power | 150W                                      | 162W                                       |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Operating System         | Ubuntu 20.04.4 LTS                        | Ubuntu 20.04.4 LTS                         |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Kernel Version           | 5.4.0-107-generic                         | 5.4.0-107-generic                          |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | BIOS Vendor              | AMI                                       | American Megatrends Inc.                   |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | BIOS Version & Release   | 4.1.16; date: 06/23/2020                  | 4.1.16; date: 06/23/2020                   |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Docker Version           | 20.10.3                                   | 20.10.3                                    |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Network Speed            | 40 Gb/s                                   | 40 Gb/s                                    |
         +--------------------------+-------------------------------------------+--------------------------------------------+

.. dropdown:: Platform with Intel® Core™ i9-10920X

      .. table:: 
         :widths: 25 25 50

         +--------------------------+-------------------------------------------+--------------------------------------------+
         |                          | Server Platform                           | Client Platform                            |
         +==========================+===========================================+============================================+
         | Motherboard              | ASUSTeK COMPUTER INC. PRIME X299-A II     | ASUSTeK COMPUTER INC. PRIME Z370-P         |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Memory                   | Corsair 4 x 16GB @ 2666 MT/s DDR4         | Corsair 4 x 16GB @ 2133 MT/s DDR4          |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | CPU                      | Intel® Core™ i9-10920X CPU @ 3.50GHz      | Intel® Core™ i7-8700T CPU @ 2.40GHz        |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Selected CPU Flags       | Hyper Threading, Turbo Boost, DL Boost    | Hyper Threading, Turbo Boost, DL Boost     |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | CPU Thermal Design Power | 165W                                      | 35 W                                       |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Operating System         | Ubuntu 20.04.4 LTS                        | Ubuntu 20.04.4 LTS                         |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Kernel Version           | 5.4.0-107-generic                         | 5.4.0-107-generic                          |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | BIOS Vendor              | American Megatrends Inc.                  | American Megatrends Inc.                   |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | BIOS Version & Release   | 0702; date: 06/10/2020                    | 2401; date: 07/15/2019                     |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Docker Version           | 19.03.13                                  | 19.03.14                                   |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Network Speed            | 10 Gb/s                                   | 10 Gb/s                                    |
         +--------------------------+-------------------------------------------+--------------------------------------------+
  

.. dropdown:: Platform with Intel® Core™ i7-8700T

      .. table:: 
         :widths: 25 25 50

         +--------------------------+-------------------------------------------+--------------------------------------------+
         |                          | Server Platform                           | Client Platform                            |
         +==========================+===========================================+============================================+
         | Motherboard              | ASUSTeK COMPUTER INC. PRIME Z370-P        | ASUSTeK COMPUTER INC. PRIME X299-A II      |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Memory                   | Corsair 4 x 16GB @ 2133 MT/s DDR4         | Corsair 4 x 16GB @ 2666 MT/s DDR4          |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | CPU                      | Intel® Core™ i7-8700T CPU @ 2.40GHz       | Intel® Core™ i9-10920X CPU @ 3.50GHz       |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Selected CPU Flags       | Hyper Threading, Turbo Boost              | Hyper Threading, Turbo Boost               |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | CPU Thermal Design Power | 35W                                       | 165 W                                      |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Operating System         | Ubuntu 20.04.4 LTS                        | Ubuntu 20.04.4 LTS                         |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Kernel Version           | 5.4.0-107-generic                         | 5.4.0-107-generic                          |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | BIOS Vendor              | American Megatrends Inc.                  | American Megatrends Inc.                   |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | BIOS Version & Release   | 2401; date: 07/15/2019                    | 0702; date: 06/10/2020                     |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Docker Version           | 19.03.14                                  | 19.03.13                                   |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Network Speed            | 10 Gb/s                                   | 10 Gb/s                                    |
         +--------------------------+-------------------------------------------+--------------------------------------------+

.. dropdown:: Platform with Intel® Core™ i5-8500

      .. table:: 
         :widths: 25 25 50

         +--------------------------+-------------------------------------------+--------------------------------------------+
         |                          | Server Platform                           | Client Platform                            |
         +==========================+===========================================+============================================+
         | Motherboard              | ASUSTeK COMPUTER INC. PRIME Z370-A        | Gigabyte Technology Co., Ltd. Z390 UD      |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Memory                   | Corsair 2 x 16GB @ 2133 MT/s DDR4         | 029E 4 x 8GB @ 2400 MT/s DDR4              |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | CPU                      | Intel® Core™ i5-8500 CPU @ 3.00GHz        | Intel® Core™ i3-8100 CPU @ 3.60GHz         |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Selected CPU Flags       | Turbo Boost                               |                                            |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | CPU Thermal Design Power | 65W                                       | 65 W                                       |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Operating System         | Ubuntu 20.04.4 LTS                        | Ubuntu 20.04.1 LTS                         |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Kernel Version           | 5.4.0-113-generic                         | 5.4.0-52-generic                           |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | BIOS Vendor              | American Megatrends Inc.                  | American Megatrends Inc.                   |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | BIOS Version & Release   | 3004; date: 07/12/2021                    | F10j; date: 09/16/2020                     |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Docker Version           | 19.03.13                                  | 20.10.0                                    |
         +--------------------------+-------------------------------------------+--------------------------------------------+
         | Network Speed            | 40 Gb/s                                   | 40 Gb/s                                    |
         +--------------------------+-------------------------------------------+--------------------------------------------+

@endsphinxdirective