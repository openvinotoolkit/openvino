# Changing Input Shapes {#openvino_docs_OV_UG_ShapeInference}

This guide presents information about operating on Input Shapes, divided into C++ and Python sections.

## Introduction (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

OpenVINO™ provides capabilities to change model input shape during the runtime.
It may be useful when you want to feed model an input that has different size than model input shape. 
In case you need to do this only once prepare a model with updated shapes via [Model Optimizer](@ref when_to_specify_input_shapes). For all the other cases, follow further instructions.

### Set a New Input Shape with Reshape Method

The `ov::Model::reshape` method updates input shapes and propagates them down to the outputs of the model through all intermediate layers.
For example, changing the batch size and spatial dimensions of input of a model with an image input:

![shape_inference_explained](./img/original_vs_reshaped_model.png)

Consider the code below to achieve that:

@snippet snippets/ShapeInference.cpp picture_snippet

### Set a New Batch Size with set_batch Method

Meaning of the model batch may vary depending on the model design.
In order to change the batch dimension of the model, [set the ov::Layout](@ref declare_model_s_layout) and call the `ov::set_batch` method.

@snippet snippets/ShapeInference.cpp set_batch

The `ov::set_batch` method is a high level API of the `ov::Model::reshape` functionality, so all information about the `ov::Model::reshape` method implications are applicable for the `ov::set_batch` too, including the troubleshooting section.

Once the input shape of the `ov::Model` is set, call the `ov::Core::compile_model` method to get an `ov::CompiledModel` object for inference with updated shapes.

There are other approaches to change model input shapes during the stage of [IR generation](@ref when_to_specify_input_shapes) or [ov::Model creation](../OV_Runtime_UG/model_representation.md).

### Dynamic Shape Notice

Shape-changing functionality could be used to turn dynamic model input into a static one and vice versa.
It is recommended to always set static shapes when the shape of data is not going to change from one inference to another.
Setting static shapes avoids possible functional limitations, memory and run time overheads for dynamic shapes that vary depending on hardware plugin and model used.
To learn more about dynamic shapes in OpenVINO, see the [dedicated article](../OV_Runtime_UG/ov_dynamic_shapes.md).

### Usage of the Reshape Method <a name="usage_of_reshape_method"></a>

The primary method of the feature is `ov::Model::reshape`. It is overloaded to better serve two main use cases:

1) To change input shape of model with a single input you may pass a new shape into the method. See the example of adjusting spatial dimensions to the input image:

@snippet snippets/ShapeInference.cpp spatial_reshape

To do the opposite - resize input image to the input shapes of the model, use the [pre-processing API](../OV_Runtime_UG/preprocessing_overview.md).

2) Otherwise, you can express reshape plan via mapping of input and its new shape:
* `map<ov::Output<ov::Node>, ov::PartialShape` specifies input by passing actual input port
* `map<size_t, ov::PartialShape>` specifies input by its index
* `map<string, ov::PartialShape>` specifies input by its name

@sphinxdirective

.. tab:: Port

    .. doxygensnippet:: docs/snippets/ShapeInference.cpp
       :language: cpp
       :fragment: [obj_to_shape]

.. tab:: Index

    .. doxygensnippet:: docs/snippets/ShapeInference.cpp
       :language: cpp
       :fragment: [idx_to_shape]

.. tab:: Tensor Name

    .. doxygensnippet:: docs/snippets/ShapeInference.cpp
       :language: cpp
       :fragment: [name_to_shape]

@endsphinxdirective

Usage scenarios of the `reshape` feature can be found in the [samples section](Samples_Overview.md), starting with the [Hello Reshape Sample](../../samples/cpp/hello_reshape_ssd/README.md).

In practice, some models are not ready to be reshaped. In such cases, a new input shape cannot be set with the Model Optimizer or the `ov::Model::reshape` method.

@anchor troubleshooting_reshape_errors
### Troubleshooting Reshape Errors

Operation semantics may impose restrictions on input shapes of the operation. 
Shape collision during shape propagation may be a sign that a new shape does not satisfy the restrictions. 
Changing the model input shape may result in intermediate operations shape collision.

Examples of such operations:
* [Reshape](../ops/shape/Reshape_1.md) operation with a hard-coded output shape value.
* [MatMul](../ops/matrix/MatMul_1.md) operation with the `Const` second input cannot be resized by spatial dimensions due to operation semantics.

Model structure and logic should not change significantly after model reshaping.
- The Global Pooling operation is commonly used to reduce output feature map of classification models output.
Having the input of the shape [N, C, H, W], Global Pooling returns the output of the shape [N, C, 1, 1].
Model architects usually express Global Pooling with the help of the `Pooling` operation with the fixed kernel size [H, W].
During spatial reshape, having the input of the shape [N, C, H1, W1], Pooling with the fixed kernel size [H, W] returns the output of the shape [N, C, H2, W2], where H2 and W2 are commonly not equal to `1`.
It breaks the classification model structure.
For example, publicly available [Inception family models from TensorFlow](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) have this issue.

- Changing the model input shape may significantly affect its accuracy.
For example, Object Detection models from TensorFlow have resizing restrictions by design. 
To keep the model valid after the reshape, choose a new input shape that satisfies conditions listed in the `pipeline.config` file. 
For details, refer to the [Tensorflow Object Detection API models resizing techniques](@ref custom-input-shape).

@anchor how-to-fix-non-reshape-able-model
### How To Fix Non-Reshape-able Model

Some operators which prevent normal shape propagation can be fixed. To do so you can:
* see if the issue can be fixed via changing the values of some operators input. 
For example, most common problem of non-reshape-able models is a `Reshape` operator with hardcoded output shape.
You can cut-off hard-coded 2nd input of `Reshape` and fill it in with relaxed values.
For the following example on the picture Model Optimizer CLI should be:
```sh
mo --input_model path/to/model --input data[8,3,224,224],1:reshaped[2]->[0 -1]`
```
With `1:reshaped[2]` you request to cut 2nd input (counting from zero, so `1:` means 2nd inputs) of operation named `reshaped` and replace it with a `Parameter` with shape `[2]`.
With `->[0 -1]` you replace this new `Parameter` by a `Constant` operator which has value `[0, -1]`. 
Since `Reshape` operator has `0` and `-1` as a specific values (see the meaning in [the specification](../ops/shape/Reshape_1.md)) it allows propagating shapes freely without losing the intended meaning of `Reshape`.

![batch_relaxed](./img/batch_relaxation.png)

* transform model during Model Optimizer conversion on the back phase. For more information, see the [Model Optimizer extension](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md).
* transform OpenVINO Model during the runtime. For more information, see the [OpenVINO Runtime Transformations](../Extensibility_UG/ov_transformations.md).
* modify the original model with the help of original framework.

### Extensibility
OpenVINO provides a special mechanism that allows adding support of shape inference for custom operations. This mechanism is described in the [Extensibility documentation](../Extensibility_UG/Intro.md)

## Introduction (Python)

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

OpenVINO™ provides capabilities to change model input shape during the runtime.
It may be useful when you want to feed model an input that has different size than model input shape. 
In case you need to do this only once [prepare a model with updated shapes via Model Optimizer](@ref when_to_specify_input_shapes) for all the other cases follow further instructions.

### Set a New Input Shape with Reshape Method

The [Model.reshape](api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape) method updates input shapes and propagates them down to the outputs of the model through all intermediate layers.
Example: Changing the batch size and spatial dimensions of input of a model with an image input:

![shape_inference_explained](./img/original_vs_reshaped_model.png)

Consider the code below to achieve that:

@sphinxdirective

.. doxygensnippet:: docs/snippets/ShapeInference.py
   :language: python
   :fragment: [picture_snippet]
 
@endsphinxdirective

### Set a New Batch Size with set_batch Method

Meaning of the model batch may vary depending on the model design.
In order to change the batch dimension of the model, [set the layout](@ref declare_model_s_layout) for inputs and call the [set_batch](api/ie_python_api/_autosummary/openvino.runtime.set_batch.html) method.

@sphinxdirective

.. doxygensnippet:: docs/snippets/ShapeInference.py
   :language: python
   :fragment: [set_batch]
 
@endsphinxdirective

[set_batch](api/ie_python_api/_autosummary/openvino.runtime.set_batch.html) method is a high level API of [Model.reshape](api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape) functionality, so all information about [Model.reshape](api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape) method implications are applicable for [set_batch](api/ie_python_api/_autosummary/openvino.runtime.set_batch.html) too, including the troubleshooting section.

Once the input shape of [Model](api/ie_python_api/_autosummary/openvino.runtime.Model.html) is set, call the [compile_model](api/ie_python_api/_autosummary/openvino.runtime.compile_model.html) method to get a [CompiledModel](api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html) object for inference with updated shapes.

There are other approaches to change model input shapes during the stage of [IR generation](@ref when_to_specify_input_shapes) or [Model creation](../OV_Runtime_UG/model_representation.md).

### Dynamic Shape Notice

Shape-changing functionality could be used to turn dynamic model input into a static one and vice versa.
It is recommended to always set static shapes when the shape of data is not going to change from one inference to another.
Setting static shapes avoids possible functional limitations, memory and run-time overheads for dynamic shapes that vary depending on hardware plugin and used model.
To learn more about dynamic shapes in OpenVINO, see the [dedicated article](../OV_Runtime_UG/ov_dynamic_shapes.md).

### Usage of the Reshape Method <a name="usage_of_reshape_method"></a>

The primary method of the feature is [Model.reshape](api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape). It is overloaded to better serve two main use cases:

1) To change input shape of a model with a single input you may pass a new shape into the method. See the example of adjusting spatial dimensions to the input image:

@sphinxdirective

.. doxygensnippet:: docs/snippets/ShapeInference.py
   :language: python
   :fragment: [simple_spatials_change]
 
@endsphinxdirective

To do the opposite - resize input image to the input shapes of the model, use the [pre-processing API](../OV_Runtime_UG/preprocessing_overview.md).

2) Otherwise, you can express reshape plan via dictionary mapping input and its new shape:
Dictionary keys could be:
* The `str` specifies input by its name.
* The `int` specifies input by its index.
* The `openvino.runtime.Output` specifies input by passing actual input object.

Dictionary values (representing new shapes) could be:
* `list`
* `tuple`
* `PartialShape`

@sphinxdirective

.. tab:: Port

    .. doxygensnippet:: docs/snippets/ShapeInference.py
       :language: python
       :fragment: [obj_to_shape]

.. tab:: Index

    .. doxygensnippet:: docs/snippets/ShapeInference.py
       :language: python
       :fragment: [idx_to_shape]

.. tab:: Tensor Name

    .. doxygensnippet:: docs/snippets/ShapeInference.py
       :language: python
       :fragment: [name_to_shape]

@endsphinxdirective

Find usage scenarios of the `reshape` feature in the [samples](Samples_Overview.md), starting with [Hello Reshape Sample](../../samples/python/hello_reshape_ssd/README.md).

In practice, some models are not ready to be reshaped. In such cases, a new input shape cannot be set with the Model Optimizer or the `Model.reshape` method.

### Troubleshooting Reshape Errors

Operation semantics may impose restrictions on input shapes of the operation. 
Shape collision during shape propagation may be a sign that a new shape does not satisfy the restrictions. 
Changing the model input shape may result in intermediate operations shape collision.

Examples of such operations:
* [Reshape](../ops/shape/Reshape_1.md) operation with a hard-coded output shape value
* [MatMul](../ops/matrix/MatMul_1.md) operation with the `Const` second input cannot be resized by spatial dimensions due to operation semantics

Model structure and logic should not change significantly after model reshaping.
- The Global Pooling operation is commonly used to reduce output feature map of classification models output.
Having the input of the shape [N, C, H, W], Global Pooling returns the output of the shape [N, C, 1, 1].
Model architects usually express Global Pooling with the help of the `Pooling` operation with the fixed kernel size [H, W].
During spatial reshape, having the input of the shape [N, C, H1, W1], Pooling with the fixed kernel size [H, W] returns the output of the shape [N, C, H2, W2], where H2 and W2 are commonly not equal to `1`.
It breaks the classification model structure.
For example, [publicly available Inception family models from TensorFlow](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) have this issue.

- Changing the model input shape may significantly affect its accuracy.
For example, Object Detection models from TensorFlow have resizing restrictions by design. 
To keep the model valid after the reshape, choose a new input shape that satisfies conditions listed in the `pipeline.config` file. 
For details, refer to the [Tensorflow Object Detection API models resizing techniques](@ref custom-input-shape).

### How To Fix Non-Reshape-able Model

Some operators which prevent normal shape propagation can be fixed. To do so you can:
* see if the issue can be fixed via changing the values of some operators input. 
E.g., most common problem of non-reshape-able models is a `Reshape` operator with hardcoded output shape.
You can cut-off hard-coded 2nd input of `Reshape` and fill it in with relaxed values.
For the following example on the picture Model Optimizer CLI should be:
```sh
mo --input_model path/to/model --input data[8,3,224,224],1:reshaped[2]->[0 -1]`
```
With `1:reshaped[2]` you request to cut 2nd input (counting from zero, so `1:` means 2nd inputs) of operation named `reshaped` and replace it with a `Parameter` with shape `[2]`.
With `->[0 -1]` you replace this new `Parameter` by a `Constant` operator which has value `[0, -1]`. 
Since `Reshape` operator has `0` and `-1` as a specific values (see the meaning in [the specification](../ops/shape/Reshape_1.md)) it allows propagating shapes freely without losing the intended meaning of `Reshape`.

![batch_relaxed](./img/batch_relaxation.png)

* transform model during Model Optimizer conversion on the back phase. See the [Model Optimizer extension article](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md)
* transform OpenVINO Model during the runtime. See the [OpenVINO Runtime Transformations article](../Extensibility_UG/ov_transformations.md)
* modify the original model with the help of original framework

### Extensibility
OpenVINO provides a special mechanism that allows adding support of shape inference for custom operations. This mechanism is described in the [Extensibility documentation](../Extensibility_UG/Intro.md)
