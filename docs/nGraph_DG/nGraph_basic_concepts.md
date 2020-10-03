# nGraph Basic Concepts {#openvino_docs_nGraph_DG_basic_concepts}

The nGraph represents neural networks in uniform format. User can create different operations and combined their to one `ngraph::Function`.

## nGraph Function and Graph Representation <a name="ngraph_function"></a>

nGraph function is a very simple thing: it stores shared pointers to `ngraph::op::Result` and `ngraph::op::Parameter` operations that are inputs and outputs of the graph. 
All other operations hold each other via shared pointers: child operation holds its parent (hard link). If operation has no consumers and it's not Result operation
(shared pointer counter is zero) then it will be destructed and won't be accessible anymore. Each operation in `ngraph::Function` has a `std::shared_ptr<ngraph::Node>` type.

For details on how to build an nGraph Function, see the [Build nGraph Function](./build_function.md) page.

## Operations

`ngraph::Op` represents any abstract operations in the nGraph representation. You need to use this class to create [custom operations](../IE_DG/Extensibility_DG/AddingNGraphOps.md).

## Operation Sets

Operation set represents the set of some nGraph operations. `nGraph::Opset` is a class which provide a functionality to work with operation sets.
Custom operation set should be created to support custom operation. Please read [Extensibility DevGuide](../IE_DG/Extensibility_DG/Intro.md) for more details.

## Static and Partial Shapes

nGraph has two types for shape representation: 
`ngraph::Shape` - represents static shape.
`ngraph::PartialShape` - represents dynamic shape. That means that rank or some of dimensions are dynamic (undefined).
`ngraph::PartialShape` can be converted to `ngraph::Shape` using `get_shape()` method if all dimensions are static otherwise conversion will raise an exception.

* `ngraph::Shape` represents the static (fully defined) shapes.
  TODO: Need examples (Copy from nGraph doc and ./nGraphTransformation.md)
  @snippet example_ngraph_utils.cpp ngraph:shape

* `ngraph::PartialShape` defines a class to work with dynamic shapes.
  TODO: Need examples (Copy from nGraph doc and ./nGraphTransformation.md)


## Transformation API
nGraph transformation API allows you to manipulate the graph represented by `nGraph::Function`. For more details, see the [Overview of Transformations API](./nGraphTransformation.md) section.

## Pattern Matcher

For more details, see the Pattern Matching section in [Overview of Transformations API](./nGraphTransformation.md) section.
TODO: Need examples (Copy from nGraph doc and ./nGraphTransformation.md)
