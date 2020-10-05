# Build a Model with nGraph Library {#openvino_docs_IE_DG_nGraphTutorial}

This section illustrates how to construct an nGraph function 
composed of operations from the `opset3` namespace. Once created, 
it can wrap into a `CNNNetwork`, creating utility for data scientists 
or app developers to define a deep-learning model in a neutral way
that does not depend on existing Deep Learning (DL) frameworks.

Operation Set `opsetX` integrates a list of nGraph pre-compiled operations that work
for this purpose. In other words, `opsetX` defines a set of operations for building a graph.

For a complete list of operation sets supported by Inference Engine, see [Available Operations Sets](../ops/opset.md).

To add custom nGraph operations to an existing `CNNNetwork`, see 
the [Add Custom nGraph Operations](Extensibility_DG/Intro.md) document.

Now that you can build graphs with anything from the `opset3` definition, some 
parameters for shape-relevant (or shape-specific) inputs can be added. The 
following code prepares a graph for shape-relevant parameters. 

> **NOTE**: `validate_nodes_and_infer_types(ops)` must be included for partial shape inference. 

@snippet openvino/docs/snippets/nGraphTutorial.cpp part0

To wrap it into a CNNNetwork, use: 

@snippet openvino/docs/snippets/nGraphTutorial.cpp part1

## See Also

* [Available Operation Sets](../ops/opset.md)
* [Operation Set `opset1` Specification](../ops/opset1.md)
* [Operation Set `opset2` Specification](../ops/opset2.md)
* [Operation Set `opset3` Specification](../ops/opset3.md)
* [Inference Engine Extensibility Developer Guide](Extensibility_DG/Intro.md)
