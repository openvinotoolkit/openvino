# Deployment Optimization Guide Additional Configurations {#openvino_docs_deployment_optimization_guide_dldt_optimization_guide_additional}

To optimize your performance results during runtime step, you can experiment with:

* multi socket CPUs

* threading

* Basic Interoperability with Other APIs


## Best Latency on the Multi-Socket CPUs
Note that when latency is of concern, there are additional tips for multi-socket systems.
When input is limited to the single image, the only way to achieve the best latency is to limit execution to the single socket.
The reason is that single image is simply not enough
to saturate more than one socket. Also NUMA overheads might dominate the execution time.
Below is the example command line that limits the execution to the single socket using numactl for the best *latency* value
(assuming the machine with 28 phys cores per socket):
```
limited to the single socket).
$ numactl -m 0 --physcpubind 0-27  benchmark_app -m <model.xml> -api sync -nthreads 28
 ```
Note that if you have more than one input, running as many inference requests as you have NUMA nodes (or sockets)
usually gives the same best latency as a single request on the single socket, but much higher throughput. Assuming two NUMA nodes machine:
```
$ benchmark_app -m <model.xml> -nstreams 2
 ```
Number of NUMA nodes on the machine can be queried via 'lscpu'.
Please see more on the NUMA support in the [Optimization Guide](../OV_Runtime_UG/supported_plugins/MULTI.md).
 
  
 ##  Threading 

 - As explained in the <a href="#cpu-checklist">CPU Checklist</a> section, by default the Inference Engine uses Intel TBB as a parallel engine. Thus, any OpenVINO-internal threading (including CPU inference) uses the same threads pool, provided by the TBB. But there are also other threads in your application, so oversubscription is possible at the application level:
- The rule of thumb is that you should try to have the overall number of active threads in your application equal to the number of cores in your machine. Keep in mind the spare core(s) that the OpenCL driver under the GPU plugin might also need.
- One specific workaround to limit the number of threads for the Inference Engine is using the [CPU configuration options](../OV_Runtime_UG/supported_plugins/CPU.md).
- To avoid further oversubscription, use the same threading model in all modules/libraries that your application uses. Notice that third party components might bring their own threading. For example, using Inference Engine which is now compiled with the TBB by default might lead to [performance troubles](https://www.threadingbuildingblocks.org/docs/help/reference/appendices/known_issues/interoperability.html) when mixed in the same app with another computationally-intensive library, but compiled with OpenMP. You can try to compile the [open source version](https://github.com/opencv/dldt) of the Inference Engine to use the OpenMP as well. But notice that in general, the TBB offers much better composability, than other threading solutions.
- If your code (or third party libraries) uses GNU OpenMP, the Intel&reg; OpenMP (if you have recompiled Inference Engine with that) must be initialized first. This can be achieved by linking your application with the Intel OpenMP instead of GNU OpenMP, or using `LD_PRELOAD` on Linux* OS.

## Basic Interoperability with Other APIs <a name="basic-interoperability-with-other-apis"></a>

The general approach for sharing data between Inference Engine and media/graphics APIs like Intel&reg; Media Server Studio (Intel&reg; MSS) is based on sharing the *system* memory.  That is, in your code, you should map or copy the data from the API to the CPU address space first.

For Intel MSS, it is recommended to perform a viable pre-processing, for example, crop/resize, and then convert to RGB again with the [Video Processing Procedures (VPP)](https://software.intel.com/en-us/node/696108). Then lock the result and create an Inference Engine blob on top of that. The resulting pointer can be used for the `SetBlob`:

@snippet snippets/dldt_optimization_guide2.cpp part2

**WARNING**: The `InferenceEngine::NHWC` layout is not supported natively by most InferenceEngine plugins so internal conversion might happen.

@snippet snippets/dldt_optimization_guide3.cpp part3

Alternatively, you can use RGBP (planar RGB) output from Intel MSS. This allows to wrap the (locked) result as regular NCHW which is generally friendly for most plugins (unlike NHWC). Then you can use it with `SetBlob` just like in previous example:

@snippet snippets/dldt_optimization_guide4.cpp part4

The only downside of this approach is that VPP conversion to RGBP is not hardware accelerated (and performed on the GPU EUs). Also, it is available only on LInux.

## OpenCV* Interoperability Example <a name="opencv-interoperability"></a>

Unlike APIs that use dedicated address space and/or special data layouts (for instance, compressed OpenGL* textures), regular OpenCV data objects like `cv::Mat` reside in the conventional system memory. That is, the memory can be actually shared with the Inference Engine and only data ownership to be transferred.

Again, if the OpenCV and Inference Engine layouts match, the data can be wrapped as Inference Engine (input/output) blob. Notice that by default, Inference Engine accepts the **planar** and **not interleaved** inputs in NCHW, so the NHWC (which is exactly the interleaved layout) should be specified explicitly:

**WARNING**: The `InferenceEngine::NHWC` layout is not supported natively by most InferenceEngine plugins so internal conversion might happen.

@snippet snippets/dldt_optimization_guide5.cpp part5

Notice that original `cv::Mat`/blobs cannot be used simultaneously by the application and the Inference Engine. Alternatively, the data that the pointer references to can be copied to unlock the original data and return ownership to the original API.

To learn more about optimizations during developing step, visit [Deployment Optimization Guide](dldt_deployment_optimization_guide.md) page.
