# OpenVINO Model Representation {#openvino_docs_OV_Runtime_UG_Model_Representation}

The OpenVINO Model represents neural networks in uniform format. User can create different operations and combined their to one `ov::Model`.

## Model Representation

OpenVINO Model is a very simple thing: it stores shared pointers to `ov::op::v0::Parameter`, `ov::op::v0::Result` and `ov::op::Sink` operations that are inputs, outputs and sinks of the graph.
Sinks of the graph have no consumers and are not included in the results vector. All other operations hold each other via shared pointers: child operation holds its parent (hard link). If operation has no consumers and it's not Result or Sink operation
(shared pointer counter is zero) then it will be destructed and won't be accessible anymore. Each operation in `ov::Model` has a `std::shared_ptr<ov::Node>` type.

For details on how to build an OpenVINO Model, see the [Build nGraph Function](./build_function.md) page.

## Operations

`ov::Op` represents any abstract operations in the OpenVINO Model representation. You need to use this class to create [custom operations](../OV_Runtime_UG/Extensibility_DG/AddingNGraphOps.md).

## Operation Sets

Operation set represents the set of some OpenVINO operations. `ov::OpSet` is a class which provide a functionality to work with operation sets.
OpenVINO provides separate namespace for each operation set.

## Static and Partial Shapes

OpenVINO has two types for shape representation: 

* `ov::Shape` - Represents static (fully defined) shapes.

* `ov::PartialShape` - Represents dynamic shapes. That means that the rank or some of dimensions are dynamic (undefined). `ov::PartialShape` can be converted to `ov::Shape` using the `get_shape()` method if all dimensions are static; otherwise the conversion raises an exception.


## Build OpenVINO Model from source

OpenVINO Model can be created from source, this section illustrates how to construct an OpenVINO Model composed of operations from an available opset.

Operation Set `opsetX` integrates a list of OpenVINO pre-compiled operations that work for this purpose. In other words, `opsetX` defines a set of operations for building a graph.

For a complete list of operation sets supported by OpenVINO, see [Available Operations Sets](../ops/opset.md).

To add suport of custom operations, see the [Add Custom OpenVINO Operations](../OV_Runtime_UG/Extensibility_DG/Intro.md) document.

To build `ov::Model` from `opset8` operations please include next files:

@sphinxdirective

.. tab:: C++

   .. code-block:: cpp

      @snippet example_ngraph_utils.cpp ov:include

.. tab:: Python

   .. code-block:: python

      import openvino.runtime.opset8 as ov
      from openvino.runtime import Model

@endsphinxdirective

Below an example of simple OpenVINO Model

@sphinxdirective

.. tab:: C++

   .. code-block:: cpp

      @snippet example_ngraph_utils.cpp ov:create_simple_model

.. tab:: Python

   .. code-block:: python

      TBD

@endsphinxdirective

Example of OpenVINO Model with several outputs

@sphinxdirective

.. tab:: C++

   .. code-block:: cpp

      @snippet example_ngraph_utils.cpp ov:create_advanced_model

.. tab:: Python

   .. code-block:: python

      TBD

@endsphinxdirective

## FAQ

 - Does OpenVINO have any capabilities to debug the Model structure and Model modification?
   - To receive additional messages about applied graph modifications, rebuild the OpenVINO runtime library with the `-DENABLE_OPENVINO_DEBUG=ON` option.
   - OpenVINO Model can be visualized to image from xDot format:
    @snippet example_ngraph_utils.cpp ov:visualize
   - OpenVINO Model can be serialized to IR:
    @snippet example_ngraph_utils.cpp ov:serialize
 - How can I develop my own transformation pass?
   - Please take a look to [Transformations Developer Guide](./nGraphTransformation.md)

## See Also

* [Available Operation Sets](../ops/opset.md)
* [Operation Set `opset1` Specification](../ops/opset1.md)
* [Operation Set `opset2` Specification](../ops/opset2.md)
* [Operation Set `opset3` Specification](../ops/opset3.md)
* [Operation Set `opset4` Specification](../ops/opset4.md)
* [Operation Set `opset5` Specification](../ops/opset5.md)
* [Operation Set `opset6` Specification](../ops/opset6.md)
* [Operation Set `opset7` Specification](../ops/opset7.md)
* [Operation Set `opset8` Specification](../ops/opset8.md)
* [Inference Engine Extensibility Developer Guide](../OV_Runtime_UG/Extensibility_DG/Intro.md)
