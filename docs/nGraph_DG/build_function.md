# Build nGraph Function {#openvino_docs_nGraph_DG_build_function}

This section illustrates how to construct an nGraph function 
composed of operations from an available opset. Once created, 
it can wrap into a `CNNNetwork`, creating utility for data scientists 
or app developers to define a deep-learning model in a neutral way
that does not depend on existing Deep Learning (DL) frameworks.

Operation Set `opsetX` integrates a list of nGraph pre-compiled operations that work
for this purpose. In other words, `opsetX` defines a set of operations for building a graph.

For a complete list of operation sets supported by Inference Engine, see [Available Operations Sets](../ops/opset.md).

To add custom nGraph operations to an existing `CNNNetwork`, see 
the [Add Custom nGraph Operations](../IE_DG/Extensibility_DG/Intro.md) document.

Below you can find examples on to how build `ngraph::Function` from the `opset3` operations:

@snippet example_ngraph_utils.cpp ngraph:include

@snippet example_ngraph_utils.cpp ngraph_utils:simple_function

@snippet example_ngraph_utils.cpp ngraph_utils:advanced_function

To wrap it into a CNNNetwork, use: 
```cpp
CNNNetwork net (ng_function);
```
## See Also

* [Available Operation Sets](../ops/opset.md)
* [Operation Set `opset1` Specification](../ops/opset1.md)
* [Operation Set `opset2` Specification](../ops/opset2.md)
* [Operation Set `opset3` Specification](../ops/opset3.md)
* [Operation Set `opset4` Specification](../ops/opset4.md)
* [Inference Engine Extensibility Developer Guide](../IE_DG/Extensibility_DG/Intro.md)
