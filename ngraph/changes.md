# API Changes

## Deprecation Notice

<table>
  <tr>
    <td><strong>Deprecation Begins</strong></td>
    <td>June 1, 2020</td>
  </tr>
  <tr>
    <td><strong>Removal Date</strong></td>
    <td>December 1, 2020</td>
  </tr>
</table> 

*Starting with the OpenVINO™ toolkit 2020.2 release, all of the features previously available through nGraph have been merged into the OpenVINO™ toolkit. As a result, all the features previously available through ONNX RT Execution Provider for nGraph have been merged with ONNX RT Execution Provider for OpenVINO™ toolkit.*

*Therefore, ONNX RT Execution Provider for nGraph will be deprecated starting June 1, 2020 and will be completely removed on December 1, 2020. Users are recommended to migrate to the ONNX RT Execution Provider for OpenVINO™ toolkit as the unified solution for all AI inferencing on Intel® hardware.*

## Op Definition
* Every Op class must declare a `static constexpr NodeTypeInfo type_info{name, version}` in the class definition and define it in the .cpp file. See any op definition for an example.
* The boolean function `is_type<T>` is for testing if a node is the op `T`.
* `T as_type_ptr<T>()` and `T as_type<T>()` will upcast `Node` to an explicit op class if it is of class `T`, or `nullptr` if it is not.

## Backend library interface
* Each backend `BACKEND` needs to define the macro `${BACKEND}_API` appropriately to import symbols
  referenced from outside the library and to export them from within the library. See any
  of the `${backend}_backend_visibility.hpp` files for an example. 
* The `CMakeLists.txt` file for a backend defines `${BACKEND}_BACKEND_DLL_EXPORTS`.
  `target_compile_definitions(${backend}_backend PRIVATE ${BACKEND}_BACKEND_DLL_EXPORTS)`
* Each backend must define a function `ngraph_register_${backend}_backend` that registers a
  backend constructor function and ensures that initializations are performed.
  `ngraph/src/runtime/cpu/cpu_backend.cpp` has an example that includes initializations.
  Remove the old backend constructor code.

## Passes
* `LikeReplacement` pass must be run by all transformers.
* `ngraph::pass::FusionType` is now an enum class. Constant values defined by `FusionType` are created for backward compatibility and will be removed in future releases.

## Nodes, Parameters

* `Nodes` is now `NodeVector`
* `Parameters` is now `ParameterVector`
* `NodeVector`, `ParameterVector`, `AxisVector`, `AxisSet`, `Shape`, `Stride`, `Coordinate`, and `CoordinateDiff` are now classes, not type aliases.
* `PrimaryTensorView` is now `TensorView` (and will merge into `Tensor`)
* `copy_with_new_args` is protected; use `copy_with_new_inputs` which takes an `OutputVector` as an argument and preserves control dependencies.

## Changes to ops

* The namespace `ngraph::op` is only for actual ops. Helpers have been moved into
  `ngraph::op::util`:
  + `BinaryElementwiseArithmetic`
  + `BinaryElementwiseComparison`
  + `BinaryElementwise`
  + `RequiresTensorViewArgs`
  + `UnaryElementwiseArithmetic`
  + `UnaryElementwise`
  Ops defined outside of nGraph core will need to get the base class from `ngraph::op::util` and
  change the include file to `#include "ngraph/ops/util/requires_tensor_view_args.hpp"`, etc.

  See any of the core ops for an example.

## Changes to convolution and pooling ops

* Backprop ops have been added for convolution ops.
* The convolution and pooling ops have had several methods/fields renamed, to reflect a shift
  in terminology from "images" to "data". Generally this just means that you will have to
  `s/image_batch/data_batch/` and `s/image_dilation_strides/data_dilation_strides/`.
* The following functions have been removed:
  + `AvgPool`: `get_channel_count get_input_image_physical_shape get_input_image_virtual_shape get_output_image_shape get_batch_size get_image_dimension_count`
  + `MaxPool`: `get_channel_count get_input_image_shape get_output_image_shape get_batch_size get_image_dimension_count`
  + `Convolution`: `get_input_channel_count get_output_channel_count get_input_image_physical_shape get_input_image_virtual_shape get_output_image_shape get_window_physical_shape get_window_virtual_shape get_batch_size get_image_dimension_count`

  All of the above information can be inferred from the shapes and parameters of the op.

* The `AvgPool` operator has a new attribute governing whether or not padding-region values
  are considered when computing a given window's average: `include_padding_in_avg_computation`.
  One of the class constructors adds this to the parameter list, and the others use a default
  value of `false` which matches the old behavior.

## Negative convolution padding

`Convolution` now allows negative padding. This means that the `padding_below` and `padding_above`
arguments now take type `CoordinateDiff` instead of `Shape`. `CoordinateDiff` is an alias for
`std::vector<std::ptrdiff_t>`, which "is like `size_t` but is allowed to be negative". Callers may
need to be adapted.

## Changes to Concat op	

* `get_concatenation_axis` was renamed to `get_axis`. In order to provide backward compatibility `get_concatenation_axis` is now alis of `get_axis` method	
* `set_concatenation_axis` was renamed to `set_axis`. In order to provide backward compatibility `set_concatenation_axis` is now alis of `set_axis` method

## `Parameter` and `Function` no longer take a type argument.

## Changes to Tensor read and write methods

The `read` and `write` methods on ngraph::runtime::Tensor which take a `tensor_offset` as the
second of three arguments have been deprecated. The replacement `read` and `write` methods take
two arguments, the buffer pointer and the size. For any references to the deprecated methods
remove the second argument, the tensor offset, to update to the new API. These old read/write
methods have been decorated with deprecated warnings which may be enabled by setting
`-DNGRAPH_DEPRECATED_ENABLE=ON`.

To update, remove the passed argument. For example,
```C++
// Old
make_shared<Parameter>(make_shared<descriptor::TensorViewType>(element::f32, Shape{2, 4}));
// New (remove TensorViewType)
make_shared<Parameter>(element::f32, Shape{2, 4});

// Old
make_shared<Function>(results, result_type, parameters);
// New
make_shared<Function>(results, parameters);
```

The runtime::Tensor methods to get_tensor<> and write<T>(std::vector&) have been removed
to the unit test directory under utils/test_tool.hpp read_vector and write_vector.

## Changes to reshape op utils

Utility functions from `src/ngraph/op/util/reshape.hpp`, placed at namespace `ngraph::op::util`:

  - `reshape`
  - `reorder_axes`
  - `transpose`
  - `flatten`

Are moved to new location: `src/ngraph/builder/reshape.hpp` to namespace `ngraph::builder`.
