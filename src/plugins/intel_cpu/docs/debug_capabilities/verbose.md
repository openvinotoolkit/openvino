# Verbose mode

It is possible to enable tracing execution of plugin nodes to cout and collect statistics, such as:
  - inference request number
  - node implementer:
    * cpu (CPU plugin)
    * dnnl (oneDNN library)
    * ngraph_ref (ngraph reference fallback)
  - node name
  - node type
  - node algorithm
  - node primitive info
  - input / output ports info
  - fused nodes (omitted if no)
  - time measurements:
    * for static case - total
    * for dynamic case - total, shapeInfer, prepareParams, exec
  - cache hit info (omitted if no):
    * execCacheHit (if primitives cache hit occurred)
  - etc

Format:
```sh
    ov_cpu_verbose,<inference_request_number>,exec,<node_implemeter>,
    <node_name>:<node_type>:<node_alg>,<impl_type>,
    src:<port_id>:<precision>::<type>:<format>:f0:<shape> ...
    dst:<port_id>:<precision>::<type>:<format>:f0:<shape> ...,
    post_ops:'<node_name>:<node_type>:<node_alg>;...;',
    time:<time_measurement>:<number>ms ...,
    cacheHit:<cache_hit_info> ...
```

To turn on verbose mode the following environment variable should be used:
```sh
    OV_CPU_VERBOSE=<level> binary ...
```

Currently verbose mode has only one level, any digit can be used for activation.

To have colored verbose output just duplicate level's digit, for example:
```sh
    OV_CPU_VERBOSE=11 binary ...
```
**NOTE:** Shell color codes are used
