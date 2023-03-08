# CPU target for SnippetS code generator

Snippets in its first generation can be seen as a generalization over generic eltwise node. First generation of snippets has lack of integration with oneDNN and so patterns it supports should be kept orthogonal to what is fused with post-ops. 

POC CPU implementation could be found [here](https://github.com/openvinotoolkit/openvino/pull/2824)

First 8 kernel parameters are passed by structure which is unpacked inside a kernel into the registers. The rest are passed through the stack.

Loop trip count should be placed to some GP register, as well as work amount. Moreover, we need to load all the parameters into GP registers. If we assume that we have enough registers than it can be done before the loop body.

```
auto param0 = abi_params[0];
auto param1 = abi_params[1];
auto result = abi_params[2];

auto work_amount = abi_params[3];
```

## Memory operations

Load could be Vector, Scalar and Broadcast. Only native vector size for an architecture is supported (e.g. 16 on AVX-512)

Memory operation also generates post increments for the pointer it uses. 

- `MemoryEmitter`
    - `StoreEmitter`
    - `ScalarStoreEmitter`
    - `LoadEmitter` (post increment)
    - `BroadcastLoadEmitter`
    - `ScalarLoadEmitter` (post increment)

## Tensor blocking

All inputs and outputs should be the same layout. Re-layout operations are not included in the snippets dialect. Since current scope is limited to layout-oblivious operations no specific handling for blocking is required. Extending dialect with re-layout operations is a subject of further benchmarking. The following memory representation is assumed.

```
 offset              domain                margin
+-------+-------------------------------+----------+
|       |                               |          |
|       |                               |          |
|       |                               |          |
|       |                               |          |
+-------+-------------------------------+----------+
```

Tensor data can be passed with strides.

## Data section

`Data` corresponds to a constant table and wraps this entity for the CPU.

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO SnippetS](../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
 