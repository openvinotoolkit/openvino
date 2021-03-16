# Performance Benchmarks {#openvino_docs_performance_benchmarks}

The [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) helps accelerate deep learning inference across a variety of Intel® processors and accelerators.  

The benchmarks below demonstrate high performance gains on several public neural networks on multiple Intel® CPUs and VPUs covering a broad performance range. Use this data to help you decide which hardware is best for your applications and solutions, or to plan your AI workload on the Intel computing already included in your solutions. 

Use the links below to review the benchmarking results for each alternative: 

* [Intel® Distribution of OpenVINO™ toolkit Benchmark Results](performance_benchmarks_openvino.md)  
* [OpenVINO™ Model Server Benchmark Results](performance_benchmarks_ovms.md)

Measuring inference performance involves many variables and is extremely use-case and application dependent. We use the below four parameters for measurements, which are key elements to consider for a successful deep learning inference application:

- **Throughput** - Measures the number of inferences delivered within a latency threshold. (for example, number of Frames Per Second - FPS). When deploying a system with deep learning inference, select the throughput that delivers the best trade-off between latency and power for the price and performance that meets your requirements.
- **Value** - While throughput is important, what is more critical in edge AI deployments is the performance efficiency or performance-per-cost. Application performance in throughput per dollar of system cost is the best measure of value.
- **Efficiency** - System power is a key consideration from the edge to the data center. When selecting deep learning solutions, power efficiency (throughput/watt) is a critical factor to consider. Intel designs provide excellent power efficiency for running deep learning workloads.
- **Latency** - This measures the synchronous execution of inference requests and is reported in milliseconds. Each inference request (for example: preprocess, infer, postprocess) is allowed to complete before the next is started. This performance metric is relevant in usage scenarios where a single image input needs to be acted upon as soon as possible. An example would be the healthcare sector where medical personnel only request analysis of a single ultra sound scanning image or in real-time or near real-time applications for example an industrial robot's response to actions in its environment or obstacle avoidance for autonomous vehicles.   

