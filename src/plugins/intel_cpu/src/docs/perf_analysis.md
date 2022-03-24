# Performance analysis step by step

## Introduction
Model performance analysis is the process of measuring the CPU plugin performance for a particular DL model inference, 
identifying possible performance hotspots, and looking for possible performance improvements.
Model performance analysis is a powerful tool for CPU plugin development since it provides valuable input on the current
performance bottlenecks, which, in turn, outlines the directions of performance improvements. Though the information on 
sub-optimal performance is obtained on specific models, usually many neural networks have common architectural elements,
which means performance tuning for a specific model leads to the improvement of many similar topologies at once.

## Objectives
- Understand the current state of the CPU plugin performance
- Detect possible performance optimizations
- Evaluate the impact of proposed optimizations to the model performance

## Step by step guide
1. Fix the CPU frequency, close all background tasks that may affect performance.
2. Run the model with `benchmark_app` using one stream for some defined time interval to avoid potential performance
   deviations (get average estimation) with the average perf counters and execution graph dumping:
   ```shell script
   ./benchmark_app -m <path_to_model>/<model_name>.xml -hint latency -t 10 -exec_graph_path <path_to_exec_graph>/<exec_graph_name>.xml --report_type average_counters --report_folder <path_to_report_folder> -pc
   ```
   See [benhamrk tool documentation](https://docs.openvino.ai/latest/openvino_inference_engine_samples_benchmark_app_README.html) for details.
   Latency mode is used by default because it gives performance results that are easy to interpret since only one stream per CPU socket is created and
   all the infer requests are executed sequentially.
3. Make an outcome table summarizing the average time per each layer type.
   One can either process raw perf counters output or use the plugin's debug capabilities (see [performance measurements guide](https://github.com/openvinotoolkit/openvino/blob/b257ca1cca3f942f612993ce869f7f730d250ba8/src/plugins/intel_cpu/src/docs/perf_measurements.md) for details).
4. Analyze performance of nodes implementations:
    1. Start from node types that take the most significant amount of the execution time.
    2. For memory bound ops use the execution graph to compare the execution time with the tensor size.
       Compare the execution time with Eltwise and Reorder nodes running with the same tensor size.
       These operations give rough time estimation for memory bound ops with *linear* complexity.
       But in some cases, these ops may be not well optimized, e.g., if the channel size is not evenly divided by the SIMD vector length the Eltwise node performance is not optimal.
       Another approach to estimate peak data transfer efficiency is to replace the existing implementation with `memcpy` call and remeasure the execution time.
       Use algorithm complexity analysis to extrapolate the time estimates of a linear algorithm to the complexity of the operation being analyzed.
    3. For compute bound ops (e.g. Convolution, GEMM, LSTM) make sure the most performant implementation type is selected (e.g., for convolutions JIT implementations are preferred over GEMM).
       One can compute efficiency level using ratio between theoretical performance of the CPU, operation complexity in GFLOPs and the measured execution time.
    4. If the execution time looks suspiciously high, it makes sense to analyze the implementation for possible optimizations.
5. Analyze the execution graph in order to optimize the model on topological level:
    1. Pay attention to Reorders.
        1. Can they be avoided through more optimal tensor layout selection? Sometimes an implementation does not support
           a particular layout, or provides sub-optimal performance with the desired layout under the current circumstances. In such
           a situation, there is little that can be done to avoid reorder inserting.
           But there are situations where the implementation can perform well enough and all we need to do is tweak the layout selection algorithm to select the same layout as
           the parent or child node to avoid extra reordering operations.
        2. Can reorders be avoided through the support of various tensor precisions in node implementations? Some node implementations
           do not support the full set of plugin's native precisions, and as a result, reorders may be inserted to convert the input
           precision to that supported by the implementation. If the implementation can be extended to include additional precisions support
           without a performance penalty, it will help avoid inserting of extra reorder layers.
    2. Pay attention to simple math ops:
        1. Can we transform the graph so that the ops may be fused? To get more information about nodes fusing see
           [plugin performance optimizations guide](https://github.com/openvinotoolkit/openvino/wiki/Internal-CPU-Plugin-Optimizations).
        2. Is there a pattern that can be mapped to operations from the opset? Original DL models some times express an DL operation
           as a composition of simple one. Model optimizer and the runtime try to detect such compositions and match them with high-level
           operations from the [OpenVINO™ operation set](https://docs.openvino.ai/latest/openvino_docs_ops_opset.html) that have highly optimized implementations in plugins.
           <details>
           <summary>See MVN pattern example</summary>
           
           ![mvn_pattern](./img/mvn_pattern.png)
           
           </details>
           But some times the mapping does not properly work due to deviations from the programmed pattern.
           During the analysis we need to detect such situations to adjust the pattern matching mechanism.
    3. Look at data movement ops:
        1. Is there an opposite operation nearby? There could be situations when a chain of tensor transformation ops has
           result with the same tensor as the initial one.
           <details>
           <summary>See example with two mutually opposite transpose nodes</summary>
           
           ![double_transpose](./img/double_transpose.png)
           
           </details>
        2. Can the in-place memory usage be applied? Some operations may be performed without transferring data from the
           input tensor to the output, i.e., in-place. In such conditions either only input or output memory is used.
        3. Is there a pattern that can be mapped to an existing high-level operation? The same as it is described in the
           previous item, some groups of operations can be mapped to one high-level op.
    4. Look at low precision propagation (please see [low precision optimization guide](https://docs.openvino.ai/latest/pot_docs_LowPrecisionOptimizationGuide.html#doxid-pot-docs-low-precision-optimization-guide) for details):
        1. Whether the layers can be executed in low precision? The main idea is to place quantization and dequatization
           operation so that as many operations as possible are executed in low precision without negative impact on the model accuracy.
6. Look for possible performance optimizations:
    1. To solve the problem of sub-optimal node performance, analyze the current solution from algorithm complexity and code implementation standpoints.
       Consider using directly optimized JIT generated code as an ultimate solution.
    2. To deal with optimizations at the topological level consider managing the existing ngraph level transformations and developing custom ones to solve specific problems.
    3. To deal with optimizations at the topological level that might not be resolved at the ngraph level (i.e., node fusing, reorders dropping, etc.), one can introduce CPU graph transformations.
    4. Revise primitive descriptors selection algorithm implementation in order to eliminate extra reorders.
7. Evaluate performance gain for proposed performance improvements:
    1. Fusing a node means considerable reduction of the I/O overhead.
       Depending on the target node implementation, for example directly optimized convolutions, such overhead may be reduced to zero.
       But other algorithms such as GEMM still requires some I/O ops to perform the fused operation.
       Considering the reduction in data flow and the fact that the computation operations of the fused node still persist, one can estimate possible performance gain.
       For memory bound ops (simple math elemetwise operations) it can be equal to the fused node execution time.
    2. Applying vectorization usually means a speed up of the order of the SIMD vector length, but such an estimation is a theoretical upper bound and should be adjusted considering I/O performance and the algorithm specifics.
    3. Use algorithm complexity analysis to evaluate execution time gain using linear copy performance as a basis for memory bound ops.
    4. Lower precision theoretically means proportional execution time reduction for memory bound ops and can outline perf gain for compute bound ops if the specific instructions such as VNNI extension can be engaged.






