# nGraph basic concepts {#openvino_docs_nGraph_DG_basic_concepts}

The nGraph represents neural networks in uniform format. User can create different operations and combined their to one `ngraph::Function`.

## ngraph::Function and graph representation <a name="ngraph_function"></a>

nGraph function is a very simple thing: it stores shared pointers to `ngraph::op::Result` and `ngraph::op::Parameter` operations that are inputs and outputs of the graph. 
All other operations hold each other via shared pointers: child operation holds its parent (hard link). If operation has no consumers and it's not Result operation
(shared pointer counter is zero) then it will be destructed and won't be accessible anymore. Each operation in `ngraph::Function` has a `std::shared_ptr<ngraph::Node>` type.

Below you can find examples how `ngraph::Function` can be created:

@snippet example_ngraph_utils.cpp ngraph_utils:simple_function

@snippet example_ngraph_utils.cpp ngraph_utils:advanced_function

## Operation set

Operation set represents the set of some nGraph operations. `nGraph::Opset` is a class which provide a functionality to work with operation sets.
Custom operation set should be created to support custom operation. Please read [Extensibility DevGuide](../IE_DG/Extensibility_DG/Intro.md) for more details.

## Operation

`ngraph::Op` represents any abstract operations in the nGraph representation. You need to use this class to create [custom operations](../IE_DG/Extensibility_DG/AddingNGraphOps.md).

## Shape

`ngraph::Shape` represents the static (fully defined) shapes.

TODO: Need examples (Copy from nGraph doc and ./nGraphTransformation.md)

## Partial Shape

`ngraph::PartialShape` defines a class to work with dynamic shapes.

TODO: Need examples (Copy from nGraph doc and ./nGraphTransformation.md)


## Transformation Pass

TODO: Need examples (Copy from nGraph doc and ./nGraphTransformation.md)

## Pattern Matcher

TODO: Need examples (Copy from nGraph doc and ./nGraphTransformation.md)
