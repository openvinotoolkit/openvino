# Intel® Distribution of OpenVINO™ toolkit Benchmark Results {#openvino_docs_performance_benchmarks_openvino}

@sphinxdirective
.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_performance_benchmarks_faq
   openvino_docs_performance_int8_vs_fp32
   Performance Data Spreadsheet (download xlsx) <https://docs.openvino.ai/2022.2/_static/benchmarks_files/OV-2022.3-Performance-Data.xlsx>

@endsphinxdirective


Click the "Benchmark Graphs" button to see the OpenVINO™ benchmark graphs. Select the models, the hardware platforms (CPU SKUs), 
precision and performance index from the lists and click the “Build Graphs” button.

@sphinxdirective


.. raw:: html

    <section class="build-benchmark-section">
    <div class="title">
        <h3>Build benchmark graphs to your specifications</h3>
    </div>
    <div class="btn-container">
        <button id="build-graphs-btn" class="configure-graphs-btn">Configure Graphs</button>
    </div>
    <img src="_static/images/sample-graph-image.png" class="sample-graph-image">
    </section>

@endsphinxdirective

Measuring inference performance involves many variables and is extremely use-case and application dependent. 
Below are four parameters for measurements, which are key elements to consider for a successful deep learning inference application:

@sphinxdirective

.. raw:: html

    <div class="picker-options">
    <span class="selectable option throughput selected" data-option="throughput">
        Throughput
    </span>
    <span class="selectable option value" data-option="value">
        Value
    </span>
    <span class="selectable option efficiency" data-option="efficiency">
        Efficiency
    </span>
    <span class="selectable option latency" data-option="latency">
        Latency
    </span>
    <p class="selectable throughput selected">
        Measures the number of inferences delivered within a latency threshold. (for example, number of Frames Per Second - FPS). When deploying a system with deep learning inference, select the throughput that delivers the best trade-off between latency and power for the price and performance that meets your requirements.
    </p>
    <p class="selectable value">
        While throughput is important, what is more critical in edge AI deployments is the performance efficiency or performance-per-cost. Application performance in throughput per dollar of system cost is the best measure of value. The value KPI is calculated as “Throughput measured as inferences per second / price of inference engine”. This means for a 2 socket system 2x the price of a CPU is used. Prices are as per date of benchmarking and sources can be found as links in the Hardware Platforms (PDF) description below.
    <p class="selectable efficiency">
        System power is a key consideration from the edge to the data center. When selecting deep learning solutions, power efficiency (throughput/watt) is a critical factor to consider. Intel designs provide excellent power efficiency for running deep learning workloads. The efficiency KPI is calculated as “Throughput measured as inferences per second / TDP of inference engine”. This means for a 2 socket system 2x the power dissipation (TDP) of a CPU is used. TDP-values are as per date of benchmarking and sources can be found as links in the Hardware Platforms (PDF) description below.
    <p class="selectable latency">
        This measures the synchronous execution of inference requests and is reported in milliseconds. Each inference request (for example: preprocess, infer, postprocess) is allowed to complete before the next is started. This performance metric is relevant in usage scenarios where a single image input needs to be acted upon as soon as possible. An example would be the healthcare sector where medical personnel only request analysis of a single ultra sound scanning image or in real-time or near real-time applications for example an industrial robot's response to actions in its environment or obstacle avoidance for autonomous vehicles.
    </p>
    </div>

    <h3>Platform & Configurations </h3>
    <p>For a listing of all platforms and configurations used for testing, refer to the following:</p>
    <container class="platform-configurations">
        <div>
        <a href="https://docs.openvino.ai/nightly/_static/benchmarks_files/platform_list_22.3.pdf" target="_blank" class="pdf"><img src="_static/css/media/pdf-icon.svg"/>Hardware Platforms (PDF)</a>
        </div>
        <div>
        <a href="https://docs.openvino.ai/nightly/_static/benchmarks_files/OV-2022.3-system-info-detailed.xlsx" class="xls"><img src="_static/css/media/xls-icon.svg"/>Configuration Details (XLSX)</a>
        </div>
    </container>

@endsphinxdirective

This benchmark setup includes a single machine on which both the benchmark application and the OpenVINO™ installation reside. The presented performance benchmark numbers are based on the release 2022.3 of the Intel® Distribution of OpenVINO™ toolkit.
The benchmark application loads the OpenVINO™ Runtime and executes inferences on the specified hardware (CPU, GPU or VPU). 
It measures the time spent on actual inferencing (excluding any pre or post processing) and then reports on the inferences per second (or Frames Per Second). 


## Disclaimers

Intel® Distribution of OpenVINO™ toolkit performance benchmark numbers are based on release 2022.3.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Learn more at intel.com, or from the OEM or retailer. Performance results are based on testing as of December 13, 2022 and may not reflect all publicly available updates. See configuration disclosure for details. No product can be absolutely secure.

Performance varies by use, configuration and other factors. Learn more at [www.intel.com/PerformanceIndex](https://www.intel.com/PerformanceIndex).

Your costs and results may vary.

Intel optimizations, for Intel compilers or other products, may not optimize to the same degree for non-Intel products.

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.