@sphinxdirective
:orphan:
@endsphinxdirective
# OpenVINO™ Model Server Benchmark Results {#openvino_docs_performance_benchmarks_ovms}


@sphinxdirective
Click the "Benchmark Graphs" button to see the OpenVINO™ benchmark graphs. Select the models, the hardware platforms (CPU SKUs), 
precision and performance index from the lists and click the “Build Graphs” button.

.. button-link:: #
   :class: ov-toolkit-benchmark-results
   :color: primary
   :outline:
   
   :material-regular:`bar_chart;1.4em` Benchmark Graphs


OpenVINO™ Model Server is an open-source, production-grade inference platform that exposes a set of models via a convenient inference API 
over gRPC or HTTP/REST. It employs the OpenVINO™ Runtime libraries from the Intel® Distribution of OpenVINO™ toolkit to extend workloads 
across Intel® hardware including CPU, GPU and others.
@endsphinxdirective


![OpenVINO™ Model Server](../img/performance_benchmarks_ovms_01.png)

## Measurement Methodology

OpenVINO™ Model Server is measured in multiple-client-single-server configuration using two hardware platforms connected by ethernet network. The network bandwidth depends on the platforms as well as models under investigation and it is set to not be a bottleneck for workload intensity. This connection is dedicated only to the performance measurements. The benchmark setup is consists of four main parts:

![OVMS Benchmark Setup Diagram](../img/performance_benchmarks_ovms_02.png)

* **OpenVINO™ Model Server** is launched as a docker container on the server platform and it listens (and answers on) requests from clients. OpenVINO™ Model Server is run on the same machine as the OpenVINO™ toolkit benchmark application in corresponding benchmarking. Models served by OpenVINO™ Model Server are located in a local file system mounted into the docker container. The OpenVINO™ Model Server instance communicates with other components via ports over a dedicated docker network.

* **Clients** are run in separated physical machine referred to as client platform. Clients are implemented in Python3 programming language based on TensorFlow* API and they work as parallel processes. Each client waits for a response from OpenVINO™ Model Server before it will send a new next request. The role played by the clients is also verification of responses.

* **Load balancer** works on the client platform in a docker container. HAProxy is used for this purpose. Its main role is counting of requests forwarded from clients to OpenVINO™ Model Server, estimating its latency, and sharing this information by Prometheus service. The reason of locating the load balancer on the client site is to simulate real life scenario that includes impact of physical network on reported metrics.

* **Execution Controller** is launched on the client platform. It is responsible for synchronization of the whole measurement process, downloading metrics from the load balancer, and presenting the final report of the execution.



@sphinxdirective



Platform & Configurations
####################################

For a listing of all platforms and configurations used for testing, refer to the following:

.. button-link:: _static/benchmarks_files/platform_list_22.3.pdf
   :color: primary
   :outline:

   :material-regular:`download;1.5em` Click for Hardware Platforms [PDF]

.. button-link:: _static/benchmarks_files/OV-2022.3-system-info-detailed.xlsx
   :color: primary
   :outline:

   :material-regular:`download;1.5em` Click for Configuration Details [XLSX]

.. the files above need to be changed to the proper ones!!!

The presented performance benchmark numbers are based on the release 2022.2 of the Intel® Distribution of OpenVINO™ toolkit.
The benchmark application loads the OpenVINO™ Runtime and executes inferences on the specified hardware (CPU, GPU or GNA). 
It measures the time spent on actual inference (excluding any pre or post processing) and then reports on the inferences per second (or Frames Per Second). 

Disclaimers
####################################

Intel® Distribution of OpenVINO™ toolkit performance benchmark numbers are based on release 2022.3.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Learn more at intel.com, or from the OEM or retailer. Performance results are based on testing as of November 16, 2022 and may not reflect all publicly available updates. See configuration disclosure for details. No product can be absolutely secure.

Performance varies by use, configuration and other factors. Learn more at `www.intel.com/PerformanceIndex <https://www.intel.com/PerformanceIndex>`__.

Your costs and results may vary.

Intel optimizations, for Intel compilers or other products, may not optimize to the same degree for non-Intel products.

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.


@endsphinxdirective
