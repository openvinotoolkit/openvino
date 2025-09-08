# Memory preallocation
## Description
The `ShapePredictor` class is responsible for gathering information about shape changes of primitives and attempting to predict the shape size for the next iteration during dynamic model execution.

The motivation of implementing of such a prediction mechanism is caused by significant time costs of memory allocation on GPU for all supported types (USM and OpenCL buffers). During the dynamic model execution, the output shape of `primitive_inst` can change in any way, and in case of an increase we have to allocate new buffer. In particular, the issue becomes especially critical in the case of monotonic shape increase (as occurs with large language models).

`ShapePredictor` object is a unique object per each `InferRequest`, it stores up to 3 the most recent shapes for each primitive. It requires to have 3 shapes for the prediction, otherwise it will not perform any shape modification and return shape unchanged. Once it has 3 or more shapes of record, it starts to predict next shape. If the record shows that the shape size increases monotonically with fixed step-size, `ShapePredictor` will return larger size enough for the next 10 execution iterations. If shape size changes unpredictably, `ShapePredictor` will return shape increased by 10 percent.

## Operation modes

The main function of `ShapePredictor` is `predict_preallocation_shape()`:
```cpp
std::pair<bool, ov::Shape> predict_preallocation_shape(const std::string& id,
                                                       const ov::Shape& current_shape,
                                                       size_t dt_bitwidth,
                                                       bool can_reuse_buffer);
```

Parameters description:
* `id` refers to the `cldnn::primitive`'s unique name related to the `current_shape`
* `current_shape` describes actual shape
* `dt_bitwidth` describes buffer's data type size in bits
* `can_reuse_buffer` allows to record `current_shape` for history without applying preallocation if current buffer is enough

Return value: `predict_preallocation_shape` returns a pair of `bool` and `ov::Shape`, where `bool` value says if shape is successfully predicted and can be preallocated, and the second value is `ov::Shape` itself (empty shape will be returned if prediction is not possible).

`ShapePredictor` can operate in two modes:

* By default, it tries to predict shape for the next `_next_iters_preallocation_count` (10 by default) iterations. There are two requirements for successful shape prediction in this mode: per-iteration buffer size should be less than `_max_per_iter_size` (16KB by default) and difference between shapes' dimensions should be less than `_max_per_dim_diff`. These restrictions are needed to prevent unexpected large memory preallocations which lead to inefficient memory usage.

* The second operation mode is percentage preallocation - this mode can be configured with `ov::intel_gpu::buffers_preallocation_ratio` (1.1 by default) internal property - it increases buffer size by `_buffers_preallocation_ratio` value unconditionally if it's not possible to predict the shape for multiple iterations ahead.

Also, `ShapePredictor` has helper `can_preallocate()` function which is designed to check if desired buffer size can be preallocated or not:
```cpp
ShapePredictor::can_preallocate(size_t desired_buffer_size)
```
The flowchart describing algorithm of shape prediction:

<img src="shape_predictor_flowchart.png" width="600">
<!-- flowchart TD
    A[Store shape information]
    A -- > B{Is current buffer enough to store required data?}
    B -- >|No| C{Is the 3 shapes information collected for primitive?}
    B -- >|Yes| D
    C -- >|No| D[Return shape unchanged]
    C -- >|Yes| F{Do shapes monotonically increase?}
    F -- >|No| G[Apply percentage preallocation]
    F -- >|Yes| H{Is the shape difference within acceptable limit?}
    H -- >|No| G
    H -- >|Yes| I{Is the per-iteration buffer size within acceptable limit?}
    I -- >|No| G
    I -- >|Yes| J[Apply iterations ahead preallocation] -->


## `ShapePredictor` usage:
* Inputs/outputs preallocation (src/plugins/intel_gpu/src/plugin/sync_infer_request.cpp):
```cpp
ov::Shape predict_shape(const std::string& name,
                        const ov::Shape current_shape,
                        ov::element::Type element_type,
                        cldnn::ShapePredictor& shape_predictor) {
    // Request prediction for `current_shape` and `element_type` data type
    auto prealloc_info = shape_predictor.predict_preallocation_shape(name, current_shape, element_type.bitwidth(), false);
    const auto& preallocation_shape = prealloc_info.second;
    // Check if shape was successfully predicted and there is enough free memory for preallocation
    auto can_preallocate_buffer = prealloc_info.first &&
                                  shape_predictor.can_preallocate(cldnn::ceil_div(ov::shape_size(preallocation_shape) * element_type.bitwidth(), 8));
    if (can_preallocate_buffer) {
        return preallocation_shape;
    }

    return current_shape;
}
```
* Nodes' output buffer preallocation (src/plugins/intel_gpu/src/graph/primitive_inst.cpp):
```cpp
auto current_shape = actual_layout.get_shape();
auto& sp = *get_network().get_shape_predictor();
auto dt_size = ov::element::Type(actual_layout.data_type).bitwidth();
// Request prediction for `current_shape` and `actual_layout.data_type` data type
auto prealloc_info = sp.predict_preallocation_shape(id(), current_shape, dt_size, can_reuse_buffer);
// Check if shape was successfully predicted and there is enough free memory for preallocation
if (prealloc_info.first && sp.can_preallocate(ov::shape_size(prealloc_info.second) * dt_size)) {
    auto new_layout = actual_layout;
    new_layout.set_partial_shape(prealloc_info.second);
    // Update `updated_params` output layout which will be used for memory allocation
    updated_params.output_layouts[0] = new_layout;
}
```

## `ShapePredictor` debug capabilities:
You can use `OV_GPU_MemPreallocationOptions` environment variable (in case of enabled `DEBUG_CAPS`, see details in src/plugins/intel_gpu/docs/gpu_debug_utils.md) to change buffer preallocation behaviour. This property expects 4 values separated by space in the following order: number of iterations for preallocation(int), max size of single iteration in bytes(int), max per-dim allowed diff(int), unconditional buffers preallocation ratio(float). For example, for disabling memory preallocation at all, you can use `OV_GPU_MemPreallocationOptions='0 0 0 1.0'`
