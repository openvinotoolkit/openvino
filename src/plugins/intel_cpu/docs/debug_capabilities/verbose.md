# Verbose mode

It is possible to enable tracing execution of plugin nodes to cout and collect statistics, such as:
  - node implementer:
    * cpu (CPU plugin)
    * dnnl (oneDNN library)
    * ngraph_ref (ngraph reference fallback)
  - node name
  - node type
  - node algorithm
  - node primitive info
  - input / output ports info
  - fused nodes
  - execution time
  - etc

Format:
```sh
    ov_cpu_verbose,exec,<node_implemeter>,\
    <node_name>:<node_type>:<node_alg>,<impl_type>,\
    src:<port_id>:<precision>::<type>:<format>:f0:<shape> ...,\
    dst:<port_id>:<precision>::<type>:<format>:f0:<shape> ...,\
    post_ops:'<node_name>:<node_type>:<node_alg>;...;',\
    <execution_time>
```

To turn on verbose mode the following environment variable should be used:
```sh
    OV_CPU_VERBOSE=<level> binary ...
```

Levels enable printing of:
  - 1 - executable nodes only
  - 2 - same as `1` plus `Input` and `Output` nodes
  - 3 - same as `2` plus constant path nodes (executed in scope of `compile_model()`)

To have colored verbose output just duplicate level's digit, for example:
```sh
    OV_CPU_VERBOSE=11 binary ...
```
**NOTE:** Shell color codes are used
