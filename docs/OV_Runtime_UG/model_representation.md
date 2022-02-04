# Model Representation in OpenVINO™ Runtime {#openvino_docs_OV_Runtime_UG_Model_Representation}

In OpenVINO™ Runtime a model is represented by the `ov::Model` class.

## Model Representation

The `ov::Model` object stores shared pointers to `ov::op::v0::Parameter`, `ov::op::v0::Result` and `ov::op::Sink` operations that are inputs, outputs and sinks of the graph.
Sinks of the graph have no consumers and are not included in the results vector. All other operations hold each other via shared pointers: child operation holds its parent (hard link). If an operation has no consumers and it's not the `Result` or `Sink` operation
(shared pointer counter is zero), then it will be destructed and won't be accessible anymore. 

Each operation in `ov::Model` has the `std::shared_ptr<ov::Node>` type.

For details on how to build a model in OpenVINO™ Runtime, see the [Build Model from Source](@ref build_model) section.

## Operations

The `ov::Op` class represents any abstract operation in the model representation. Use this class to create [custom operations](../OV_Runtime_UG/Extensibility_DG/AddingNGraphOps.md).

## Operation Sets

Operation set (opset) is a collection of operations that can be used to construct a model. The `ov::OpSet` class  provides a functionality to work with operation sets.
For each operation set, OpenVINO™ Runtime provides a separate namespace, for example  `opset8`.

## Static and Partial Shapes

There are two types used for shape representation: 

* `ov::Shape` - Represents static (fully defined) shapes.

* `ov::PartialShape` - Represents dynamic shapes. That means that the rank or some of dimensions are dynamic (undefined). `ov::PartialShape` can be converted to `ov::Shape` using the `get_shape()` method if all dimensions are static; otherwise the conversion raises an exception.


## Build a Model in OpenVINO™ Runtime {#build_model}

You can create a model from source. This section illustrates how to construct a model composed of operations from an available operation set.

Operation set `opsetX` integrates a list of pre-compiled operations that work for this purpose. In other words, `opsetX` defines a set of operations for building a graph.

For a complete list of operation sets supported by OpenVINO, see [Available Operations Sets](../ops/opset.md).

To add suport of custom operations, see the [Add Custom OpenVINO Operations](../OV_Runtime_UG/Extensibility_DG/Intro.md) document.

To build `ov::Model` from `opset8` operations please include next files:

 - C++
      @snippet example_ngraph_utils.cpp ov:include

 - Python
     ```python
      import openvino.runtime.opset8 as ov
      from openvino.runtime import Model
      ```

Below an example of simple OpenVINO Model

 - C++
      @snippet example_ngraph_utils.cpp ov:create_simple_model

 - Python
     ```python
     TBD
      ```


Example of OpenVINO Model with several outputs

 - C++
      @snippet example_ngraph_utils.cpp ov:create_advanced_model

 - Python
     ```python
     TBD
      ```

## FAQ

 - Does OpenVINO have any capabilities to debug the Model structure and Model modification?
   - To receive additional messages about applied graph modifications, rebuild the OpenVINO runtime library with the `-DENABLE_OPENVINO_DEBUG=ON` option.
   - A model can be visualized to image from the xDot format:
    @snippet example_ngraph_utils.cpp ov:visualize
   - A model can be serialized to IR:
    @snippet example_ngraph_utils.cpp ov:serialize
### How can I develop my own transformation pass?
   See the [Transformations Developer Guide](./nGraphTransformation.md).

## See Also

* [Available Operation Sets](../ops/opset.md)
* [opset1 Specification](../ops/opset1.md)
* [opset2 Specification](../ops/opset2.md)
* [opset3 Specification](../ops/opset3.md)
* [opset4 Specification](../ops/opset4.md)
* [opset5 Specification](../ops/opset5.md)
* [opset6 Specification](../ops/opset6.md)
* [opset7 Specification](../ops/opset7.md)
* [opset8 Specification](../ops/opset8.md)
* [OpenVINO™ Runtime Extensibility Developer Guide](../OV_Runtime_UG/Extensibility_DG/Intro.md)
