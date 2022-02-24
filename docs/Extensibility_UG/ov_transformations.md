# Overview of Transformations API {#openvino_docs_transformations}

This guide contains all necessary information that you need to start implementing OpenVINO™ transformations.

## Transformations types <a name="transformations_types"></a>

OpenVINO™ Runtime has three main transformation types:

* `ov::pass::ModelPass` - straightforward way to work with `ov::Model` directly
* `ov::pass::MatcherPass` - pattern-based transformation approach
* `ov::pass::GraphRewrite` - container for matcher passes needed for efficient execution

![transformations_structure]

### ov::pass::ModelPass <a name="model_pass"></a>

`ov::pass::ModelPass` is used for transformations that take entire `ov::Model` as an input and process it.

Template for FunctionPass transformation class

@snippet src/transformations/template_model_transformation.hpp model_pass:template_transformation_hpp

@snippet src/transformations/template_model_transformation.cpp model_pass:template_transformation_cpp

Using `ov::pass::ModelPass`, you need to override the `run_on_model` method where you will write the transformation code.
Return value is `true` if the original function has changed during transformation (new operation was added, or operations replacement was made, or node attributes were changed); otherwise, it is `false`.
Also `ov::pass::ModelPass` based transformations can be executed via `ov::pass::Manager`. See the examples in the [Using pass manager](#using_pass_manager) section.

### ov::pass::MatcherPass <a name="matcher_pass"></a>

`ov::pass::MatcherPass` is used for pattern-based transformations.

Template for MatcherPass transformation class
@snippet src/transformations/template_pattern_transformation.hpp graph_rewrite:template_transformation_hpp

@snippet src/transformations/template_pattern_transformation.cpp graph_rewrite:template_transformation_cpp

To use `ov::pass::MatcherPass`, you need to complete these steps:
1. Create a pattern
2. Implement a callback
3. Register the pattern and Matcher
4. Execute MatcherPass

So let's go through each of these steps.

### Create a pattern

Pattern is a single root `ov::Model`. But the only difference is that you do not need to create a function object, you just need to create and connect opset or special pattern operations.
Then you need to take the last created operation and put it as a root of the pattern. This root node will be used as a root node in pattern matching.
> **NOTE**: Any nodes in a pattern that have no consumers and are not registered as root will not be used in pattern matching.

@snippet ov_model_snippets.cpp pattern:simple_example

The `Parameter` operation in the example above has type and shape specified. These attributes are needed only to create Parameter operation class and will not be used in pattern matching.

For more pattern examples, refer to the [pattern matching](#pattern_matching) section.

### Implement callback

Callback is an action applied to every pattern entrance. In general, callback is the lambda function that takes Matcher object with detected subgraph.

@snippet ov_model_snippets.cpp pattern:callback_example

The example above shows the callback structure and how Matcher can be used for accessing nodes detected by pattern.
Callback return value is `true` if root node was replaced and another pattern cannot be applied to the same root node; otherwise, it is `false`.
> **NOTE**: It is not recommended to manipulate with nodes that are under root node. This may affect GraphRewrite execution as it is expected that all nodes that come after root node in topological order are valid and can be used in pattern matching.

MatcherPass also provides functionality that allows reporting of the newly created nodes that can be used in additional pattern matching.
If MatcherPass was registered in `ov::pass::Manager` or `ov::pass::GraphRewrite`, these registered nodes will be added for additional pattern matching.
That means that matcher passes registered in `ov::pass::GraphRewrite` will be applied to these nodes.

The example below shows how single MatcherPass can fuse sequence of operations using the `register_new_node` method.

@snippet src/transformations/template_pattern_transformation.cpp matcher_pass:relu_fusion

> **NOTE**: If you register multiple nodes, please add them in topological order. We do not topologically sort these nodes as it is a time-consuming operation.

### Register pattern and Matcher

The last step is to register Matcher and callback inside the MatcherPass pass. To do this, call the `register_matcher` method.
> **NOTE**: Only one matcher can be registered for a single MatcherPass class.

```cpp
// Register matcher and callback
register_matcher(m, callback);
```
### Execute MatcherPass
MatcherPass has multiple ways to be executed:
* Run on a single node - it can be useful if you want to run MatcherPass inside another transformation.
@snippet src/transformations/template_pattern_transformation.cpp matcher_pass:run_on_node
* Run on `ov::Model` using GraphRewrite - this approach gives ability to run MatcherPass on whole `ov::Model`. Moreover, multiple MatcherPass transformation can be registered in a single GraphRewite to be executed in a single graph traversal.
@snippet src/transformations/template_pattern_transformation.cpp matcher_pass:graph_rewrite
* Run on `ov::Model` using `ov::pass::Manager` - this approach helps you to register MatcherPass for execution on `ov::Model` as another transformation types.
@snippet src/transformations/template_pattern_transformation.cpp matcher_pass:manager


### ov::pass::GraphRewrite <a name="graph_rewrite_pass"></a>

GraphRewrite pass serves for running multiple matcher passes on `ov::Model` in a single graph traversal.
Example:

@snippet src/transformations/template_pattern_transformation.cpp matcher_pass:graph_rewrite

In addition, GraphRewrite handles nodes that were registered by MatcherPasses during their execution. This nodes will be added to the beginning of the sequence with nodes for pattern matching.

> **NOTE**: when using `ov::pass::Manager` temporary GraphRewrite is used to execute single MatcherPass.

GraphRewrite has two algorithms for MatcherPasses execution. First algorithm is straightforward. It applies each MatcherPass in registration order to current node.

![graph_rewrite_execution]

But it is not really efficient when you have a lot of registered passes. So first of all GraphRewrite checks that all MatcherPass patterns has type-based root node (it means that type of this node is not hidden into predicate).
And then creates map from registered MatcherPasses. That helps to avoid additional cost of applying each MatcherPass for each node.

![graph_rewrite_efficient_search]

> **NOTE**: GraphRewrite execution algorithm cannot be set manually and depends only on root nodes registered inside MatcherPasses.

## Pattern Matching <a name="pattern_matching"></a>

Sometimes patterns cannot be expressed via regular operations or it is too complicated.
For example, if you want to detect Convolution->Add sub-graph without specifying particular input type for Convolution operation or you want to create a pattern where some of operations can have different types.
And for these cases OpenVINO™ provides additional helpers to construct patterns for GraphRewrite transformations.

There are two main helpers:
1. `ov::pass::pattern::any_input` - helps to express inputs if their types are undefined.
2. `ov::pass::pattern::wrap_type<T>` - helps to express nodes of pattern without specifying node attributes.

Let's go through the example to have better understanding of how it works:

> **NOTE**: Node attributes do not participate in pattern matching and are needed only for operations creation. Only operation types participate in pattern matching.

The example below shows basic usage of `ov::passpattern::any_input`.
Here we construct Multiply pattern with arbitrary first input and Constant as a second input.
Also as Multiply is commutative operation, it does not matter in which order we set inputs (any_input/Constant or Constant/any_input) because both cases will be matched.

@snippet ov_model_snippets.cpp pattern:label_example

This example shows how we can construct a pattern when operation has arbitrary number of inputs.

@snippet ov_model_snippets.cpp pattern:concat_example

This example shows how to use predicate to construct a pattern. Also it shows how to match pattern manually on given node.

@snippet ov_model_snippets.cpp pattern:predicate_example

> **NOTE**: Be careful with manual matching because Matcher object holds matched nodes. To clear a match, use the m->clear_state() method.

## Working with ov::Function <a name="working_with_ov_model"></a>

In this chapter extends the [model representation guide](../OV_Runtime_UG/model_representation.md) and shows an API that allows us to manipulate with `ov::Model`.

### ov::Node input and output ports

First of all let's talk about `ov::Node` input/output ports. Each OpenVINO™ operation has input and output ports except cases when operation has `Parameter` or `Constant` type.

Every port belongs to its node, so using a port we can access parent node, get shape and type for particular input/output, get all consumers in case of output port, and get producer node in case of input port.
With output port we can set inputs for newly created operations.

Lets look at the code example.

@snippet ov_model_snippets.cpp ov:ports_example

### ov::Node replacement

OpenVINO™ provides two ways for node replacement: via OpenVINO™ helper function and directly via port methods. We are going to review both of them.

Let's start with OpenVINO™ helper functions. The most popular function is `ov::replace_node(old_node, new_node)`.

We will review real replacement case where Negative operation is replaced with Multiply.

![ngraph_replace_node]

@snippet ov_model_snippets.cpp ov:replace_node

`ov::replace_node` has a constraint that number of output ports for both of ops must be the same; otherwise, it raises an exception.


The alternative way to do the same replacement is the following:

@snippet ov_model_snippets.cpp ov:manual_replace

Another transformation example is insertion.

![ngraph_insert_node]

@snippet ov_model_snippets.cpp ov:insert_node

The alternative way to the insert operation is to make a node copy and use `ov::replace_node()`:

@snippet ov_model_snippets.cpp ov:insert_node_with_copy

### ov::Node elimination

Another type of node replacement is its elimination.

To eliminate operation, OpenVINO™ has special method that considers all limitations related to OpenVINO™ Runtime.

@snippet ov_model_snippets.cpp ov:eliminate_node

`ov::replace_output_update_name()` in case of successful replacement it automatically preserves friendly name and runtime info.


## Transformation conditional compilation

Transformation library has two internal macros to support conditional compilation feature.

* `MATCHER_SCOPE(region)` - allows to disable the MatcherPass if matcher isn't used. The region name should be unique. This macro creates a local variable `matcher_name` which you should use as a matcher name.
* `RUN_ON_MODEL_SCOPE(region)` - allows to disable run_on_model pass if it isn't used. The region name should be unique.

## Transformation writing essentials <a name="transformation_writing_essentials"></a>

When developing a transformation, you need to follow these transformation rules:

###1. Friendly Names

Each `ov::Node` has an unique name and a friendly name. In transformations we care only about friendly name because it represents the name from the model.
To avoid losing friendly name when replacing node with other node or subgraph, set the original friendly name to the latest node in replacing subgraph. See the example below.

@snippet ov_model_snippets.cpp ov:replace_friendly_name

In more advanced cases, when replaced operation has several outputs and we add additional consumers to its outputs, we make a decision how to set friendly name by arrangement.

###2. Runtime Info

Runtime info is a map `std::map<std::string, ov::Any>` located inside `ov::Node` class. It represents additional attributes in `ov::Node`.
These attributes can be set by users or by plugins and when executing transformation that changes `ov::Model` we need to preserve these attributes as they will not be automatically propagated.
In most cases, transformations have the following types: 1:1 (replace node with another node), 1:N (replace node with a sub-graph), N:1 (fuse sub-graph into a single node), N:M (any other transformation).
Currently, there is no mechanism that automatically detects transformation types, so we need to propagate this runtime information manually. See the examples below.

```cpp
// Replace Transpose with Reshape operation (1:1)
ov::copy_runtime_info(transpose, reshape);
```

```cpp
// Replace Div operation with Power and Multiply sub-graph (1:N)
ov::copy_runtime_info(div, {pow, mul});
```

```cpp
// Fuse Convolution with Add operation (N:1)
ov::copy_runtime_info({conv, bias}, {conv_ie});
```

```cpp
// Any other transformation that replaces one sub-graph with another sub-graph (N:M)
ov::copy_runtime_info({a, b, c}, {e, f});
```

When transformation has multiple fusions or decompositions, `ov::copy_runtime_info` must be called multiple times for each case.

> **Note**: copy_runtime_info removes rt_info from destination nodes. If you want to keep it, you need to specify them in source nodes like this: copy_runtime_info({a, b, c}, {a, b})

###3. Constant Folding

If your transformation inserts constant sub-graphs that need to be folded, do not forget to use `ov::pass::ConstantFolding()` after your transformation or call constant folding directly for operation.
The example below shows how constant subgraph can be constructed.

@snippet ov_model_snippets.cpp ov:constant_subgraph

Manual constant folding is more preferable than `ov::pass::ConstantFolding()` because it is much faster.

Below you can find an example of manual constant folding:

@snippet src/transformations/template_pattern_transformation.cpp manual_constant_folding

## Common mistakes in transformations <a name="common_mistakes"></a>

In transformation development process:

* Do not use deprecated OpenVINO™ API. Deprecated methods has the `OPENVINO_DEPRECATED` macros in its definition.
* Do not pass `shared_ptr<Node>` as an input for other node if type of node is unknown or it has multiple outputs. Use explicit output port.
* If you replace node with another node that produces different shape, remember that new shape will not be propagated until the first `validate_nodes_and_infer_types` call for `ov::Model`. If you are using `ov::pass::Manager`, it will automatically call this method after each transformation execution.
* Do not forget to call the `ov::pass::ConstantFolding` pass if your transformation creates constant subgraphs.
* Use latest OpSet if you are not developing downgrade transformation pass.
* When developing a callback for `ov::pass::MatcherPass`,  do not change nodes that come after the root node in topological order.

## Using pass manager <a name="using_pass_manager"></a>

`ov::pass::Manager` is a container class that can store the list of transformations and execute them. The main idea of this class is to have high-level representation for grouped list of transformations.
It can register and apply any [transformation types](#transformations_types) on function.
In addition, `ov::pass::Manager` has extended debug capabilities (find more information in the [how to debug transformations](#how_to_debug_transformations) section).

The example below shows basic usage of `ov::pass::Manager`

@snippet src/transformations/template_pattern_transformation.cpp matcher_pass:manager3

Another example shows how multiple matcher passes can be united into single GraphRewrite.

@snippet src/transformations/template_pattern_transformation.cpp matcher_pass:manager2

## How to debug transformations <a name="how_to_debug_transformations"></a>

If you are using `ngraph::pass::Manager` to run sequence of transformations, you can get additional debug capabilities by using the following environment variables:

```
OV_PROFILE_PASS_ENABLE=1 - enables performance measurement for each transformation and prints execution status
OV_ENABLE_VISUALIZE_TRACING=1 -  enables visualization after each transformation. By default, it saves dot and svg files.
```

> **Note**: Make sure that you have dot installed on your machine; otherwise, it will silently save only dot file without svg file.

[ngraph_replace_node]: ./img/ngraph_replace_node.png
[ngraph_insert_node]: ./img/ngraph_insert_node.png
[transformations_structure]: ./img/transformations_structure.png
[register_new_node]: ./img/register_new_node.png
[graph_rewrite_execution]: ./img/graph_rewrite_execution.png
[graph_rewrite_efficient_search]: ./img/graph_rewrite_efficient_search.png
