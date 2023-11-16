# Performance Benchmarks {#openvino_docs_performance_benchmarks}

@sphinxdirective

.. meta::
   :description: Use the benchmark results for Intel® Distribution of OpenVINO™ 
                 toolkit, that may help you decide what hardware to use or how 
                 to plan the workload.

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_performance_benchmarks_faq
   OpenVINO Accuracy <openvino_docs_performance_int8_vs_fp32>
   Performance Data Spreadsheet (download xlsx) <https://docs.openvino.ai/2023.1/_static/benchmarks_files/OV-2023.2-Performance-Data.xlsx> 
   openvino_docs_MO_DG_Getting_Performance_Numbers


This page presents benchmark results for `Intel® Distribution of OpenVINO™ toolkit <https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html>`__ 
and :doc:`OpenVINO Model Server <ovms_what_is_openvino_model_server>`, for a representative selection of public neural networks and Intel® devices.
The results may help you decide which hardware to use in your applications or plan AI workload for the hardware you have already implemented in your solutions. 
Click the buttons below to see the chosen benchmark data.

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      .. button-link:: #
         :class: ov-toolkit-benchmark-results
         :color: primary
         :outline:
         :expand:

         :material-regular:`bar_chart;1.4em` OpenVINO Benchmark Graphs

   .. grid-item::

      .. button-link:: #
         :class: ovms-toolkit-benchmark-results
         :color: primary
         :outline:
         :expand:
   
         :material-regular:`bar_chart;1.4em` OVMS Benchmark Graphs


For a successful deep learning inference application, the following four key metrics need to be considered: 

.. tab-set::

   .. tab-item:: Throughput
      :sync: throughput

      Measures the number of inferences delivered within a latency threshold 
      (for example, number of Frames Per Second - FPS). When deploying a system with 
      deep learning inference, select the throughput that delivers the best trade-off 
      between latency and power for the price and performance that meets your requirements.

   .. tab-item:: Value
      :sync: value

      While throughput is important, what is more critical in edge AI deployments is 
      the performance efficiency or performance-per-cost. Application performance in 
      throughput per dollar of system cost is the best measure of value. The value KPI is 
      calculated as “Throughput measured as inferences per second / price of inference engine”. 
      This means for a 2 socket system 2x the price of a CPU is used. Prices are as per 
      date of benchmarking and sources can be found as links in the Hardware Platforms (PDF) description below.

   .. tab-item:: Efficiency
      :sync: efficiency

      System power is a key consideration from the edge to the data center. When selecting 
      deep learning solutions, power efficiency (throughput/watt) is a critical factor to consider. 
      Intel designs provide excellent power efficiency for running deep learning workloads. 
      The efficiency KPI is calculated as “Throughput measured as inferences per second / TDP of 
      inference engine”. This means for a 2 socket system 2x the power dissipation (TDP) of a CPU is used. 
      TDP-values are as per date of benchmarking and sources can be found as links in the Hardware Platforms (PDF) description below.

   .. tab-item:: Latency
      :sync: latency

      This measures the synchronous execution of inference requests and is reported in milliseconds. 
      Each inference request (for example: preprocess, infer, postprocess) is allowed to complete before 
      the next is started. This performance metric is relevant in usage scenarios where a single image 
      input needs to be acted upon as soon as possible. An example would be the healthcare sector where 
      medical personnel only request analysis of a single ultra sound scanning image or in real-time or 
      near real-time applications for example an industrial robot's response to actions in its environment 
      or obstacle avoidance for autonomous vehicles.


Platforms, Configurations, Methodology
###########################################################

For a listing of all platforms and configurations used for testing, refer to the following:

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      .. button-link:: _static/benchmarks_files/OV-2023.2-Platform_list.pdf
         :color: primary
         :outline:
         :expand:

         :material-regular:`download;1.5em` Click for Hardware Platforms [PDF]
      
      .. button-link:: _static/benchmarks_files/OV-2023.2-system-info-detailed.xlsx
         :color: primary
         :outline:
         :expand:
         
         :material-regular:`download;1.5em` Click for Configuration Details [XLSX]


The OpenVINO benchmark setup includes a single system with OpenVINO™, as well as the benchmark application installed.
It measures the time spent on actual inference (excluding any pre or post processing) and then reports on the inferences 
per second (or Frames Per Second). 

OpenVINO™ Model Server (OVMS) employs the Intel® Distribution of OpenVINO™ toolkit runtime libraries and exposes a set of 
models via a convenient inference API over gRPC or HTTP/REST. Its benchmark results are measured with the configuration of
multiple-clients-single-server, using two hardware platforms connected by ethernet. Network bandwidth depends on both, platforms 
and models under investigation. It is set not to be a bottleneck for workload intensity. The connection is dedicated 
only to measuring performance. 

.. dropdown:: See more details about OVMS benchmark setup

   The benchmark setup for OVMS consists of four main parts:
   
   .. image:: _static/images/performance_benchmarks_ovms_02.png
      :alt: OVMS Benchmark Setup Diagram

   * **OpenVINO™ Model Server** is launched as a docker container on the server platform and it listens (and answers on) 
     requests from clients. OpenVINO™ Model Server is run on the same machine as the OpenVINO™ toolkit benchmark application 
     in corresponding benchmarking. Models served by OpenVINO™ Model Server are located in a local file system mounted into 
     the docker container. The OpenVINO™ Model Server instance communicates with other components via ports over a dedicated docker network.
   
   * **Clients** are run in separated physical machine referred to as client platform. Clients are implemented in Python3 
     programming language based on TensorFlow* API and they work as parallel processes. Each client waits for a response from OpenVINO™ 
     Model Server before it will send a new next request. The role played by the clients is also verification of responses.
   
   * **Load balancer** works on the client platform in a docker container. HAProxy is used for this purpose. Its main role is 
     counting of requests forwarded from clients to OpenVINO™ Model Server, estimating its latency, and sharing this information by 
     Prometheus service. The reason of locating the load balancer on the client site is to simulate real life scenario that includes 
     impact of physical network on reported metrics.
   
   * **Execution Controller** is launched on the client platform. It is responsible for synchronization of the whole measurement process, 
     downloading metrics from the load balancer, and presenting the final report of the execution.



Test performance yourself
####################################

You can also test performance for your system yourself, following the guide on :doc:`getting performance numbers <openvino_docs_MO_DG_Getting_Performance_Numbers>`.

Performance of a particular application can also be evaluated virtually using `Intel® DevCloud for the Edge <https://devcloud.intel.com/edge/>`__. 
It is a remote development environment with access to Intel® hardware and the latest versions of the Intel® Distribution of the OpenVINO™ Toolkit. 
To learn more about it, visit `the website <https://www.intel.com/content/www/us/en/developer/tools/devcloud/edge/overview.html>`__ 
or `create an account <https://www.intel.com/content/www/us/en/secure/forms/devcloud-enrollment/account-provisioning.html>`__.



Disclaimers
####################################

* Intel® Distribution of OpenVINO™ toolkit performance results are based on release 2023.2, as of November 15, 2023.

* OpenVINO Model Server performance results are based on release 2023.0, as of June 01, 2023.

The results may not reflect all publicly available updates. Intel technologies’ features and benefits depend on system configuration 
and may require enabled hardware, software, or service activation. Learn more at intel.com, or from the OEM or retailer.

See configuration disclosure for details. No product can be absolutely secure.
Performance varies by use, configuration and other factors. Learn more at `www.intel.com/PerformanceIndex <https://www.intel.com/PerformanceIndex>`__.
Your costs and results may vary.
Intel optimizations, for Intel compilers or other products, may not optimize to the same degree for non-Intel products.


@endsphinxdirective
