# Model Representation in OpenVINO™ Runtime {#openvino_docs_OV_Runtime_UG_Model_Representation}

In OpenVINO™ Runtime a model is represented by the `ov::Model` class.

The `ov::Model` object stores shared pointers to `ov::op::v0::Parameter`, `ov::op::v0::Result` and `ov::op::Sink` operations that are inputs, outputs and sinks of the graph.
Sinks of the graph have no consumers and are not included in the results vector. All other operations hold each other via shared pointers: child operation holds its parent (hard link). If an operation has no consumers and it's not the `Result` or `Sink` operation
(shared pointer counter is zero), then it will be destructed and won't be accessible anymore. 

Each operation in `ov::Model` has the `std::shared_ptr<ov::Node>` type.

For details on how to build a model in OpenVINO™ Runtime, see the [Build a Model in OpenVINO™ Runtime](@ref build_model) section.

OpenVINO™ Runtime allows to use tensor names or indexes to work wit model inputs/outpus. To get model input/output ports you can use `ov::Model::input()` or `ov::Model::output()` respectively.

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/src/main.cpp
       :language: cpp
       :fragment: [part2]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/src/main.py
       :language: python
       :fragment: [part2]

@endsphinxdirective

OpenVINO™ Runtime model representation uses special classes to work with model data types and shapes. For data types the `ov::element::Type` is used.

## Shapes representation

OpenVINO™ Runtime provides two types for shape representation: 

* `ov::Shape` - Represents static (fully defined) shapes.

* `ov::PartialShape` - Represents dynamic shapes. That means that the rank or some of dimensions are dynamic (undefined). `ov::PartialShape` can be converted to `ov::Shape` using the `get_shape()` method if all dimensions are static; otherwise the conversion raises an exception.
  `ov::PartialShape` can be converted to `ov::Shape` using the `get_shape()` method if all dimensions are static; otherwise, conversion raises an exception.

  @sphinxdirective

  .. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [ov:partial_shape]

  .. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_model_snippets.py
         :language: python
         :fragment: [ov:partial_shape]

  @endsphinxdirective

  But in most cases before getting static shape using `get_shape()` method, you need to check that shape is static.

## Operations

The `ov::Op` class represents any abstract operation in the model representation. Use this class to create [custom operations](../OV_Runtime_UG/Extensibility_DG/AddingNGraphOps.md).

## Operation Sets

Operation set (opset) is a collection of operations that can be used to construct a model. The `ov::OpSet` class  provides a functionality to work with operation sets.
For each operation set, OpenVINO™ Runtime provides a separate namespace, for example `opset8`.
Each OpenVINO™ Release release introduces new operations and add these operations to a new operation set. New operation sets help to introduce a new version of operations that change behavior of previous operations. Using operation sets allows you to avoid changes in your application if new operations have been introduced.

## Build a Model in OpenVINO™ Runtime {#build_model}

You can create a model from source. This section illustrates how to construct a model composed of operations from an available operation set.

Operation set `opsetX` integrates a list of pre-compiled operations that work for this purpose. In other words, `opsetX` defines a set of operations for building a graph.

For a complete list of operation sets supported in OpenVINO™ toolkit, see [Available Operations Sets](../ops/opset.md).

To add suport of custom operations, see the [Add Custom OpenVINO Operations](../OV_Runtime_UG/Extensibility_DG/Intro.md) document.

To build an `ov::Model` instance from `opset8` operations, include the following files:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
       :language: cpp
       :fragment: [ov:include]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_model_snippets.py
       :language: python
       :fragment: [import]

@endsphinxdirective

The following code demonstrates how to create a simple model:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
       :language: cpp
       :fragment: [ov:create_simple_model]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_model_snippets.py
       :language: python
       :fragment: [ov:create_simple_model]

@endsphinxdirective

The following code creates a model with several outputs:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
       :language: cpp
       :fragment: [ov:create_advanced_model]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_model_snippets.py
       :language: python
       :fragment: [ov:create_advanced_model]

@endsphinxdirective

## FAQ

### Does OpenVINO™ Runtime provide any capabilities to debug the model structure and model modification?

@sphinxdirective
.. raw:: html

    <div class="collapsible-section">
@endsphinxdirective

   - To receive additional messages about applied graph modifications, rebuild the OpenVINO™ Runtime library with the `-DENABLE_OPENVINO_DEBUG=ON` option.
   - A model can be visualized to image from the xDot format:

    @sphinxdirective

    .. tab:: C++

        .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
           :language: cpp
           :fragment: [ov:visualize]

    .. tab:: Python

        .. doxygensnippet:: docs/snippets/ov_model_snippets.py
           :language: python
           :fragment: [ov:visualize]

    @endsphinxdirective

   - A model can be serialized to IR:

    @sphinxdirective

    .. tab:: C++

        .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
           :language: cpp
           :fragment: [ov:serialize]

    .. tab:: Python

        .. doxygensnippet:: docs/snippets/ov_model_snippets.py
           :language: python
           :fragment: [ov:serialize]

    @endsphinxdirective

@sphinxdirective
.. raw:: html

    </div>
@endsphinxdirective

### How can I develop my own transformation pass?

@sphinxdirective
.. raw:: html

    <div class="collapsible-section">
@endsphinxdirective

   See the [Transformations Developer Guide](./nGraphTransformation.md).

@sphinxdirective
.. raw:: html

    </div>
@endsphinxdirective

## See Also

* [Available Operation Sets](../ops/opset.md)
* [OpenVINO™ Runtime Extensibility Developer Guide](../OV_Runtime_UG/Extensibility_DG/Intro.md)
