# OpenVINO™ Model Server Benchmark Results {#openvino_docs_performance_benchmarks_ovms}

OpenVINO™ Model Server is an open-source, production grade inference platform that exposes a set of models via a convenient inference API over gRPC or HTTP/REST. It employs the OpenVINO™ Runtime libraries from the Intel® Distribution of OpenVINO™ toolkit to extend workloads across Intel® hardware including CPU, GPU and others.

![OpenVINO™ Model Server](../img/performance_benchmarks_ovms_01.png)

## Measurement Methodology

OpenVINO™ Model Server is measured in multiple-client-single-server configuration using two hardware platforms connected by an ethernet network. The network bandwidth depends on the platforms as well as models under investigation, and it is set to not be a bottleneck for workload intensity. This connection is dedicated only to the performance measurements. The benchmark setup consists of four main parts:

![OVMS Benchmark Setup Diagram](../img/performance_benchmarks_ovms_02.png)

- **OpenVINO™ Model Server** -- It is launched as a docker container on the server platform and it listens (and answers on) requests from clients. It is run on the same machine as the OpenVINO™ toolkit benchmark application in corresponding benchmarking. Models served by it are located in a local file system mounted into the docker container. The OpenVINO™ Model Server instance communicates with other components via ports over a dedicated docker network.

- **Clients** -- This part run in separated physical machine referred to as client platform. Clients are implemented in Python3 programming language based on TensorFlow API and they work as parallel processes. Each client waits for a response from OpenVINO™ Model Server before it will send a new next request. Clients also play role of the verification of responses.

- **Load Balancer** -- It works on the client platform in a docker container by using a HAProxy. The main role of Load Balancer is counting the requests forwarded from clients to OpenVINO™ Model Server, estimating its latency, and sharing this information by Prometheus service. The reason for locating this part on the client site is to simulate a real life scenario that includes an impact of a physical network on reported metrics.

- **Execution Controller** -- It is launched on the client platform. It is responsible for synchronization of the whole measurement process, downloading metrics from Load Balancer and presenting the final report of the execution.

## resnet-50-TF (INT8)
![](../img/throughput_ovms_resnet50_int8.png)
## resnet-50-TF (FP32)
![](../img/throughput_ovms_resnet50_fp32_bs_1.png)
## googlenet-v4-TF (FP32)
![](../img/throughput_ovms_googlenet4_fp32.png)
## yolo-v3-tf (FP32)
![](../img/throughput_ovms_yolo3_fp32.png)
## yolo-v4-tf (FP32)
![](../img/throughput_ovms_yolo4_fp32.png)
## brain-tumor-segmentation-0002
![](../img/throughput_ovms_braintumorsegmentation.png)
## alexnet
![](../img/throughput_ovms_alexnet.png)
## mobilenet-v3-large-1.0-224-TF (FP32)
![](../img/throughput_ovms_mobilenet3large_fp32.png)
## deeplabv3 (FP32)
![](../img/throughput_ovms_deeplabv3_fp32.png)
## bert-small-uncased-whole-word-masking-squad-int8-0002 (INT8)
![](../img/throughput_ovms_bertsmall_int8.png)
## bert-small-uncased-whole-word-masking-squad-0002 (FP32)
![](../img/throughput_ovms_bertsmall_fp32.png)
## 3D U-Net (FP32)
![](../img/throughput_ovms_3dunet.png)

## Image Compression for Improved Throughput
OpenVINO™ Model Server supports compressed binary input data (images in JPEG and PNG formats) for vision processing models. This
feature improves overall performance on networks where the bandwidth constitutes a system bottleneck. A good example of such a use case could be wireless 5G communication, a typical 1 Gbit/sec Ethernet network or a usage scenario with many client machines issuing a high rate of inference requests to one single central OpenVINO model server. Generally, the performance improvement grows with increased compressibility of the data/image. The decompression on the server side is performed by the OpenCV library.

### Supported Image Formats for OVMS Compression

- Always supported:
  - Portable image format - `*.pbm`, `*.pgm`, `*.ppm`, `*.pxm`, `*.pnm`.
  - Radiance HDR - `*.hdr`, `*.pic`.
  - Sun rasters - `*.sr`, `*.ras`.
  - Windows bitmaps - `*.bmp`, `*.dib`.

- Limited support (please see OpenCV documentation):
  - Raster and Vector geospatial data supported by GDAL.
  - JPEG files - `*.jpeg`, `*.jpg`, `*.jpe`.
  - Portable Network Graphics - `*.png`.
  - TIFF files - `*.tiff`, `*.tif`.
  - OpenEXR Image files - `*.exr`.
  - JPEG 2000 files - `*.jp2`.
  - WebP - `*.webp`.

### googlenet-v4-tf (FP32)
![](../img/throughput_ovms_1gbps_googlenet4_fp32.png)

### resnet-50-tf (INT8)
![](../img/throughput_ovms_1gbps_resnet50_int8.png)

### resnet-50-tf (FP32)
![](../img/throughput_ovms_1gbps_resnet50_fp32.png)

## Platform Configurations

OpenVINO™ Model Server performance benchmark numbers are based on release 2021.4. Performance results are based on testing as of June 17, 2021 and may not reflect all publicly available updates.

### Platform with Intel® Xeon® Platinum 8260M

@sphinxdirective
.. raw:: html

    <table class="table">
      <tr>
        <th></th>
        <th><strong>Server Platform</strong></th>
        <th><strong>Client Platform</strong></th>
      </tr>
      <tr>
        <td><strong>Motherboard</strong></td>
        <td>Inspur YZMB-00882-104 NF5280M5</td>
        <td>Intel® Server Board S2600WF H48104-872</td>
      </tr>
      <tr>
        <td><strong>Memory</strong></td>
        <td>Samsung 16 x 16GB @ 2666 MT/s DDR4</td>
        <td>Hynix 16 x 16GB @ 2666 MT/s DDR4</td>
      </tr>
      <tr>
        <td><strong>CPU</strong></td>
        <td>Intel® Xeon® Platinum 8260M CPU @ 2.40GHz</td>
        <td>Intel® Xeon® Gold 6252 CPU @ 2.10GHz</td>
      </tr>
      <tr>
        <td><strong>Selected CPU Flags</strong></td>
        <td>Hyper Threading, Turbo Boost, DL Boost</td>
        <td>Hyper Threading, Turbo Boost, DL Boost</td>
      </tr>
      <tr>
        <td><strong>CPU Thermal Design Power</strong></td>
        <td>162 W</td>
        <td>150 W</td>
      </tr>
      <tr>
        <td><strong>Operating System</strong></td>
        <td>Ubuntu 20.04.2 LTS</td>
        <td>Ubuntu 20.04.2 LTS</td>
      </tr>
      <tr>
        <td><strong>Kernel Version</strong></td>
        <td>5.4.0-54-generic</td>
        <td>5.4.0-65-generic</td>
      </tr>
      <tr>
        <td><strong>BIOS Vendor</strong></td>
        <td>American Megatrends Inc.</td>
        <td>Intel® Corporation</td>
      </tr>
      <tr>
        <td><strong>BIOS Version & Release</strong></td>
        <td>4.1.16, date: 06/23/2020</td>
        <td>SE5C620.86B.02.01, date: 03/26/2020</td>
      </tr>
      <tr>
        <td><strong>Docker Version</strong></td>
        <td>20.10.3</td>
        <td>20.10.3</td>
      </tr>
      <tr>
        <td><strong>Network Speed</strong></td>
        <td colspan="2">40 Gb/s</td>
      </tr>
    </table>

@endsphinxdirective

### Platform with Intel® Xeon® Gold 6252

@sphinxdirective
.. raw:: html

    <table class="table">
      <tr>
        <th></th>
        <th><strong>Server Platform</strong></th>
        <th><strong>Client Platform</strong></th>
      </tr>
      <tr>
        <td><strong>Motherboard</strong></td>
        <td>Intel® Server Board S2600WF H48104-872</td>
        <td>Inspur YZMB-00882-104 NF5280M5</td>
      </tr>
      <tr>
        <td><strong>Memory</strong></td>
        <td>Hynix 16 x 16GB @ 2666 MT/s DDR4</td>
        <td>Samsung 16 x 16GB @ 2666 MT/s DDR4</td>
      </tr>
      <tr>
        <td><strong>CPU</strong></td>
        <td>Intel® Xeon® Gold 6252 CPU @ 2.10GHz</td>
        <td>Intel® Xeon® Platinum 8260M CPU @ 2.40GHz</td>
      </tr>
      <tr>
        <td><strong>Selected CPU Flags</strong></td>
        <td>Hyper Threading, Turbo Boost, DL Boost</td>
        <td>Hyper Threading, Turbo Boost, DL Boost</td>
      </tr>
      <tr>
        <td><strong>CPU Thermal Design Power</strong></td>
        <td>150 W</td>
        <td>162 W</td>
    </tr>
      <tr>
        <td><strong>Operating System</strong></td>
        <td>Ubuntu 20.04.2 LTS</td>
        <td>Ubuntu 20.04.2 LTS</td>
      </tr>
      <tr>
        <td><strong>Kernel Version</strong></td>
        <td>5.4.0-65-generic</td>
        <td>5.4.0-54-generic</td>
      </tr>
      <tr>
        <td><strong>BIOS Vendor</strong></td>
        <td>Intel® Corporation</td>
        <td>American Megatrends Inc.</td>
      </tr>
      <tr>
        <td><strong>BIOS Version and Release Date</strong></td>
        <td>SE5C620.86B.02.01, date: 03/26/2020</td>
        <td>4.1.16, date: 06/23/2020</td>
      </tr>
      <tr>
        <td><strong>Docker Version</strong></td>
        <td>20.10.3</td>
        <td>20.10.3</td>
      </tr>
      <tr>
        <td><strong>Network Speed</strong></td>
        <td colspan="2" align="center">40 Gb/s</td>
      </tr>
    </table>

@endsphinxdirective

### Platform with Intel® Core™ i9-10920X

@sphinxdirective
.. raw:: html

    <table class="table">
    <tr>
      <th></th>
      <th><strong>Server Platform</strong></th>
      <th><strong>Client Platform</strong></th>
    </tr>
    <tr>
      <td><strong>Motherboard</strong></td>
      <td>ASUSTeK COMPUTER INC. PRIME X299-A II</td>
      <td>ASUSTeK COMPUTER INC. PRIME Z370-P</td>
    </tr>
    <tr>
      <td><strong>Memory</strong></td>
      <td>Corsair 4 x 16GB @ 2666 MT/s DDR4</td>
      <td>Corsair 4 x 16GB @ 2133 MT/s DDR4</td>
    </tr>
    <tr>
      <td><strong>CPU</strong></td>
      <td>Intel® Core™ i9-10920X CPU @ 3.50GHz</td>
      <td>Intel® Core™ i7-8700T CPU @ 2.40GHz</td>
    </tr>
    <tr>
      <td><strong>Selected CPU Flags</strong></td>
      <td>Hyper Threading, Turbo Boost, DL Boost</td>
      <td>Hyper Threading, Turbo Boost</td>
    </tr>
    <tr>
      <td><strong>CPU Thermal Design Power</strong></td>
      <td>165 W</td>
      <td>35 W</td>
    </tr>
    <tr>
      <td><strong>Operating System</strong></td>
      <td>Ubuntu 20.04.1 LTS</td>
      <td>Ubuntu 20.04.1 LTS</td>
    </tr>
    <tr>
      <td><strong>Kernel Version</strong></td>
      <td>5.4.0-52-generic</td>
      <td>5.4.0-56-generic</td>
    </tr>
    <tr>
      <td><strong>BIOS Vendor</strong></td>
      <td>American Megatrends Inc.</td>
      <td>American Megatrends Inc.</td>
    </tr>
    <tr>
      <td><strong>BIOS Version and Release Date</strong></td>
      <td>0603, date: 03/05/2020</td>
      <td>2401, date: 07/15/2019</td>
    </tr>
    <tr>
      <td><strong>Docker Version</strong></td>
      <td>19.03.13</td>
      <td>19.03.14</td>
    </tr>
    </tr>
    <tr>
      <td><strong>Network Speed</strong></td>
      <td colspan="2" align="center">10 Gb/s</td>
    </tr>
    </table>

@endsphinxdirective

### Platform with Intel® Core™ i7-8700T

@sphinxdirective
.. raw:: html

    <table class="table">
    <tr>
      <th></th>
      <th><strong>Server Platform</strong></th>
      <th><strong>Client Platform</strong></th>
    </tr>
    <tr>
      <td><strong>Motherboard</strong></td>
      <td>ASUSTeK COMPUTER INC. PRIME Z370-P</td>
      <td>ASUSTeK COMPUTER INC. PRIME X299-A II</td>
    </tr>
    <tr>
      <td><strong>Memory</strong></td>
      <td>Corsair 4 x 16GB @ 2133 MT/s DDR4</td>
      <td>Corsair 4 x 16GB @ 2666 MT/s DDR4</td>
    </tr>
    <tr>
      <td><strong>CPU</strong></td>
      <td>Intel® Core™ i7-8700T CPU @ 2.40GHz</td>
      <td>Intel® Core™ i9-10920X CPU @ 3.50GHz</td>
    </tr>
    <tr>
      <td><strong>Selected CPU Flags</strong></td>
      <td>Hyper Threading, Turbo Boost</td>
      <td>Hyper Threading, Turbo Boost, DL Boost</td>
    </tr>
    <tr>
      <td><strong>CPU Thermal Design Power</strong></td>
      <td>35 W</td>
      <td>165 W</td>
    </tr>
    <tr>
      <td><strong>Operating System</strong></td>
      <td>Ubuntu 20.04.1 LTS</td>
      <td>Ubuntu 20.04.1 LTS</td>
    </tr>
    <tr>
      <td><strong>Kernel Version</strong></td>
      <td>5.4.0-56-generic</td>
      <td>5.4.0-52-generic</td>
    </tr>
    <tr>
      <td><strong>BIOS Vendor</strong></td>
      <td>American Megatrends Inc.</td>
      <td>American Megatrends Inc.</td>
    </tr>
    <tr>
      <td><strong>BIOS Version and Release Date</strong></td>
      <td>2401, date: 07/15/2019</td>
      <td>0603, date: 03/05/2020</td>
    </tr>
    <tr>
      <td><strong>Docker Version</strong></td>
      <td>19.03.14</td>
      <td>19.03.13</td>
    </tr>
    </tr>
    <tr>
      <td><strong>Network Speed</strong></td>
      <td colspan="2" align="center">10 Gb/s</td>
    </tr>
    </table>

@endsphinxdirective

### Platform with Intel® Core™ i5-8500

@sphinxdirective
.. raw:: html

    <table class="table">
    <tr>
      <th></th>
      <th><strong>Server Platform</strong></th>
      <th><strong>Client Platform</strong></th>
    </tr>
    <tr>
      <td><strong>Motherboard</strong></td>
      <td>ASUSTeK COMPUTER INC. PRIME Z370-A</td>
      <td>Gigabyte Technology Co., Ltd. Z390 UD</td>
    </tr>
    <tr>
      <td><strong>Memory</strong></td>
      <td>Corsair 2 x 16GB @ 2133 MT/s DDR4</td>
      <td>029E 4 x 8GB @ 2400 MT/s DDR4</td>
    </tr>
    <tr>
      <td><strong>CPU</strong></td>
      <td>Intel® Core™ i5-8500 CPU @ 3.00GHz</td>
      <td>Intel® Core™ i3-8100 CPU @ 3.60GHz</td>
    </tr>
    <tr>
      <td><strong>Selected CPU Flags</strong></td>
      <td>Turbo Boost</td>
      <td>-</td>
    </tr>
    <tr>
      <td><strong>CPU Thermal Design Power</strong></td>
      <td>65 W</td>
      <td>65 W</td>
    </tr>
    <tr>
      <td><strong>Operating System</strong></td>
      <td>Ubuntu 20.04.1 LTS</td>
      <td>Ubuntu 20.04.1 LTS</td>
    </tr>
    <tr>
      <td><strong>Kernel Version</strong></td>
      <td>5.4.0-52-generic</td>
      <td>5.4.0-52-generic</td>
    </tr>
    <tr>
      <td><strong>BIOS Vendor</strong></td>
      <td>American Megatrends Inc.</td>
      <td>American Megatrends Inc.</td>
    </tr>
    <tr>
      <td><strong>BIOS Version and Release Date</strong></td>
      <td>2401, date: 07/12/2019</td>
      <td>F10j, date: 09/16/2020</td>
    </tr>
    <tr>
      <td><strong>Docker Version</strong></td>
      <td>19.03.13</td>
      <td>20.10.0</td>
    </tr>
    </tr>
    <tr>
      <td><strong>Network Speed</strong></td>
      <td colspan="2" align="center">40 Gb/s</td>
    </tr>
    </table>

@endsphinxdirective

### Platform with Intel® Core™ i3-8100

@sphinxdirective
.. raw:: html

    <table class="table">
    <tr>
      <th></th>
      <th><strong>Server Platform</strong></th>
      <th><strong>Client Platform</strong></th>
    </tr>
    <tr>
      <td><strong>Motherboard</strong></td>
      <td>Gigabyte Technology Co., Ltd. Z390 UD</td>
      <td>ASUSTeK COMPUTER INC. PRIME Z370-A</td>
    </tr>
    <tr>
      <td><strong>Memory</strong></td>
      <td>029E 4 x 8GB @ 2400 MT/s DDR4</td>
      <td>Corsair 2 x 16GB @ 2133 MT/s DDR4</td>
    </tr>
    <tr>
      <td><strong>CPU</strong></td>
      <td>Intel® Core™ i3-8100 CPU @ 3.60GHz</td>
      <td>Intel® Core™ i5-8500 CPU @ 3.00GHz</td>
    </tr>
    <tr>
      <td><strong>Selected CPU Flags</strong></td>
      <td>-</td>
      <td>Turbo Boost</td>
    </tr>
    <tr>
      <td><strong>CPU Thermal Design Power</strong></td>
      <td>65 W</td>
      <td>65 W</td>
    </tr>
    <tr>
      <td><strong>Operating System</strong></td>
      <td>Ubuntu 20.04.1 LTS</td>
      <td>Ubuntu 20.04.1 LTS</td>
    </tr>
    <tr>
      <td><strong>Kernel Version</strong></td>
      <td>5.4.0-52-generic</td>
      <td>5.4.0-52-generic</td>
    </tr>
    <tr>
      <td><strong>BIOS Vendor</strong></td>
      <td>American Megatrends Inc.</td>
      <td>American Megatrends Inc.</td>
    </tr>
    <tr>
      <td><strong>BIOS Version and Release Date</strong></td>
      <td>F10j, date: 09/16/2020</td>
      <td>2401, date: 07/12/2019</td>
    </tr>
    <tr>
      <td><strong>Docker Version</strong></td>
      <td>20.10.0</td>
      <td>19.03.13</td>
    </tr>
    </tr>
    <tr>
      <td><strong>Network Speed</strong></td>
      <td colspan="2" align="center">40 Gb/s</td>
    </tr>
    </table>

@endsphinxdirective

