# Overview of Transformations API {#ngraph_transformation}

This guide contains all necessary information that you need to start implementing nGraph transformations.

## Prerequisites
Before creating a transformation, do the following:

* Make sure that there is no transformation with the same functionality in the [Transformation Library](group__ie__transformation__api.html)
* Learn how the [Transformation Library](group__ie__transformation__api.html) is structured and how transformations are organized
* Understand where to put your transformation code

### Transformation Library Structure
Transformation library is independent from Inference Engine target library named as `inference_engine_transformations`
and is located in the `inference-engine/src/transformations` directory.

Transformations root directory contains two folders:
* `ngraph_ops` - Contains internal opset operations that are common for plugins.
* `transformations` - Includes all transformations, utils, runtime info attributes, and pass managers.

All internal operations and transformations located inside the [Transformation Library](group__ie__transformation__api.html) can be used inside plugins.
All legacy operations and transformations were moved to a legacy library and are not recommended to be used.

### Transformation Flow Layers
Transformation flow in the transformation library has several layers:

1. Pass managers - Execute any type of transformations and provide additional debug capabilities.
2. Transformations - Perform a particular transformation algorithm on `ngraph::Function`.
3. Low-level functions - Take a set of nodes and perform some transformation action.
They are not mandatory and all transformation code can be located inside the transformation.
But if some transformation parts can potentially be reused in other transformations, we suggest keeping them as a separate functions.

### Location for Your Transformation Code
To decide where to store your transformation code, please follow these rules:

1. If it is a plugin-specific transformation and cannot be reused by other plugins, keep source code inside plugin.
2. If this transformation relates to opset operation conversion or optimization, keep sources inside the transformation library.

After you decide where to store your transformation code, you can start developing your own nGraph transformation.

## ngraph::Function and graph representation <a name="ngraph_function"></a>

nGraph function is a very simple thing: it stores shared pointers to `ngraph::op::Parameter`, `ngraph::op::Result` and  `ngraph::op::Sink` operations that are inputs, outputs and sinks of the graph.
Sinks of the graph have no consumers and not included into results vector. All other operations hold each other via shared pointers: child operation holds its parent (hard link). If operation has no consumers and it's not Result or Sink operation
(shared pointer counter is zero) then it will be destructed and won't be accessible anymore. Each operation in `ngraph::Function` has a `std::shared_ptr<ngraph::Node>` type.

For examples of how to build an nGraph function, see the [Build nGraph Function](./build_function.md) page.

## Transformations types <a name="transformations_types"></a>

nGraph has three main transformation types: 

* `ngraph::pass::FunctionPass` - straightforward way to work with `ngraph::Function` directly
* `ngraph::pass::MatcherPass` - pattern-based transformation approach
* `ngraph::pass::GraphRewrite` - container for matcher passes needed for efficient execution

![transformations_structure]

### ngraph::pass::FunctionPass <a name="function_pass"></a>

`ngraph::pass::FunctionPass` is used for transformations that take entire `ngraph::Function` as an input and process it.

Template for FunctionPass transformation class

@snippet src/template_function_transformation.hpp function_pass:template_transformation_hpp

@snippet src/template_function_transformation.cpp function_pass:template_transformation_cpp

Using `ngraph::FunctionPass`, you need to override the `run_on_function` method where you will write the transformation code.
Return value is `true` if the original function has changed during transformation (new operation was added, or operations replacement was made, or node attributes were changed); otherwise, it is `false`.
For transformation API, please follow the [working with ngraph::Function](#working_with_ngraph_function) section.
Also `ngraph::FunctionPass` based transformations can be executed via `pass::Manager`. See the examples in the [Using pass manager](#using_pass_manager) section.

### ngraph::pass::MatcherPass <a name="matcher_pass"></a>

`ngraph::pass::MatcherPass` is used for pattern-based transformations.

Template for MatcherPass transformation class
@snippet src/template_pattern_transformation.hpp graph_rewrite:template_transformation_hpp

@snippet src/template_pattern_transformation.cpp graph_rewrite:template_transformation_cpp

To use `ngraph::pass::MatcherPass`, you need to complete these steps:
1. Create a pattern
2. Implement a callback 
3. Register the pattern and Matcher
4. Execute MatcherPass

So let's go through each of these steps.

### Create a pattern
Pattern is a single root `ngraph::Function`. But the only difference is that you do not need to create a function object, you just need to create and connect opset or special pattern operations.
Then you need to take the last created operation and put it as a root of the pattern. This root node will be used as a root node in pattern matching.
> **NOTE**: Any nodes in a pattern that have no consumers and are not registered as root will not be used in pattern matching. 

@snippet example_ngraph_utils.cpp pattern:simple_example

The `Parameter` operation in the example above has type and shape specified. These attributes are needed only to create Parameter operation class and will not be used in pattern matching.

For more pattern examples, refer to the [pattern matching](#pattern_matching) section.

### Implement callback
Callback is an action applied to every pattern entrance. In general, callback is the lambda function that takes Matcher object with detected subgraph.

@snippet example_ngraph_utils.cpp pattern:callback_example

The example above shows the callback structure and how Matcher can be used for accessing nodes detected by pattern.
Callback return value is `true` if root node was replaced and another pattern cannot be applied to the same root node; otherwise, it is `false`.
> **NOTE**: It is not recommended to manipulate with nodes that are under root node. This may affect GraphRewrite execution as it is expected that all nodes that come after root node in topological order are valid and can be used in pattern matching. 

MatcherPass also provides functionality that allows reporting of the newly created nodes that can be used in additional pattern matching.
If MatcherPass was registered in `pass::Manager` or `pass::GraphRewrite`, these registered nodes will be added for additional pattern matching.
That means that matcher passes registered in `pass::GraphRewrite` will be applied to these nodes.

The example below shows how single MatcherPass can fuse sequence of operations using the `register_new_node` method.

@snippet src/template_pattern_transformation.cpp matcher_pass:relu_fusion

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
@snippet src/template_pattern_transformation.cpp matcher_pass:run_on_node
* Run on `ngraph::Function` using GraphRewrite - this approach gives ability to run MatcherPass on whole `ngraph::Function`. Moreover, multiple MatcherPass transformation can be registered in a single GraphRewite to be executed in a single graph traversal.
@snippet src/template_pattern_transformation.cpp matcher_pass:graph_rewrite
* Run on `ngraph::Function` using `pass::Manager` - this approach helps you to register MatcherPass for execution on `ngraph::Function` as another transformation types.
@snippet src/template_pattern_transformation.cpp matcher_pass:manager


### ngraph::pass::GraphRewrite <a name="graph_rewrite_pass"></a>

GraphRewrite pass serves for running multiple matcher passes on `ngraph::Function` in a single graph traversal.
Example:

@snippet src/template_pattern_transformation.cpp matcher_pass:graph_rewrite

In addition, GraphRewrite handles nodes that were registered by MatcherPasses during their execution. This nodes will be added to the beginning of the sequence with nodes for pattern matching.

> **NOTE**: when using `pass::Manager` temporary GraphRewrite is used to execute single MatcherPass. 

GraphRewrite has two algorithms for MatcherPasses execution. First algorithm is straightforward. It applies each MatcherPass in registration order to current node.

![graph_rewrite_execution]

But it is not really efficient when you have a lot of registered passes. So first of all GraphRewrite checks that all MatcherPass patterns has type-based root node (it means that type of this node is not hidden into predicate).
And then creates map from registered MatcherPasses. That helps to avoid additional cost of applying each MatcherPass for each node.

![graph_rewrite_efficient_search] 

> **NOTE**: GraphRewrite execution algorithm cannot be set manually and depends only on root nodes registered inside MatcherPasses.

## Pattern Matching <a name="pattern_matching"></a>

Sometimes patterns cannot be expressed via regular nGraph operations or it is too complicated.
For example, if you want to detect Convolution->Add sub-graph without specifying particular input type for Convolution operation or you want to create a pattern where some of operations can have different types.
And for these cases nGraph provides additional helpers to construct patterns for GraphRewrite transformations. 

There are two main helpers:
1. `ngraph::pattern::any_input` - helps to express inputs if their types are undefined.
2. `ngraph::pattern::wrap_type<T>` - helps to express nodes of pattern without specifying node attributes.

Let's go through the example to have better understanding of how it works:

> **NOTE**: Node attributes do not participate in pattern matching and are needed only for operations creation. Only operation types participate in pattern matching.

The example below shows basic usage of `pattern::any_input`.
Here we construct Multiply pattern with arbitrary first input and Constant as a second input. 
Also as Multiply is commutative operation, it does not matter in which order we set inputs (any_input/Constant or Constant/any_input) because both cases will be matched.

@snippet example_ngraph_utils.cpp pattern:label_example

This example shows how we can construct a pattern when operation has arbitrary number of inputs.

@snippet example_ngraph_utils.cpp pattern:concat_example

This example shows how to use predicate to construct a pattern. Also it shows how to match pattern manually on given node.

@snippet example_ngraph_utils.cpp pattern:predicate_example

> **NOTE**: Be careful with manual matching because Matcher object holds matched nodes. To clear a match, use the m->clear_state() method.

## Working with ngraph::Function <a name="working_with_ngraph_function"></a>

In this chapter we will review nGraph API that allows us to manipulate with `ngraph::Function`.

### ngraph::Node input and output ports

First of all let's talk about `ngraph::Node` input/output ports. Each nGraph operation has input and output ports except cases when operation has `Result`, `Parameter`, or `Constant` type.

Every port belongs to its node, so using a port we can access parent node, get shape and type for particular input/output, get all consumers in case of output port, and get producer node in case of input port.
With output port we can set inputs for newly created operations. 

Lets look at the code example.

@snippet example_ngraph_utils.cpp ngraph:ports_example

You may notice that we usually construct operations in this way:
```cpp
std::shared_ptr<Node> neg_const = opset1::Constant::create(sub->get_input_element_type(1), Shape{1}, {-1}));
Output<Node> data = node->input_value(0);
auto neg = std::make_shared<ngraph::opset1::Multiply>(data, neg_const);
```
In this example, the `opset3::Multiply` operation takes `Output<Node>` and `std::shared_ptr<Node>` as inputs. But the constructor takes both as `Output<Node>`. 
In this case, `std::shared_ptr<Node>` will be automatically converted to `Output<Node>` if node has exactly one output port; otherwise, conversion raises an exception.   

### ngraph::Node replacement

nGraph provides two ways for node replacement: via nGraph helper function and directly via port methods. We are going to review both of them.

Let's start with nGraph helper functions. The most popular function is `ngraph::replace_node(old_node, new_node)`.

We will review real replacement case where Negative operation is replaced with Multiply.

![ngraph_replace_node]

@snippet example_ngraph_utils.cpp ngraph:replace_node

`ngraph::replace_node` has a constraint that number of output ports for both of ops must be the same; otherwise, it raises an exception.


The alternative way to do the same replacement is the following:
```cpp
// All neg->output(0) consumers will be moved to mul->output(0) port
neg->output(0).replace(mul->output(0));
```

Another transformation example is insertion.

![ngraph_insert_node]

@snippet example_ngraph_utils.cpp ngraph:insert_node

The alternative way to the insert operation is to make a node copy and use `replace_node`:

@snippet example_ngraph_utils.cpp ngraph:insert_node_with_copy

### ngraph::Node elimination

Another type of node replacement is its elimination.

To eliminate operation, nGraph has special method that considers all limitations related to InferenceEngine.

@snippet example_ngraph_utils.cpp ngraph:eliminate_node

`replace_output_update_name` in case of successful replacement it automatically preserves friendly name and runtime info.
  

## Transformation conditional compilation 

Transformation library has two internal macros to support conditional compilation feature.

* `MATCHER_SCOPE(region)` - allows to disable the MatcherPass if matcher isn't used. The region name should be unique. This macro creates a local variable `matcher_name` which you should use as a matcher name.
* `RUN_ON_FUNCTION_SCOPE(region)` - allows to disable run_on_function pass if it isn't used. The region name should be unique.

## Transformation writing essentials <a name="transformation_writing_essentials"></a>

When developing a transformation, you need to follow these transformation rules:

###1. Operation Set (OpSet)

Use the latest version of OpSet in your transformation. An exception is op_conversion transformations, where different opsets can be used.

@snippet example_ngraph_utils.cpp ngraph:include

###2. Dynamic Shape and Rank

nGraph has two types for shape representation: 
`ngraph::Shape` - represents static shape.
`ngraph::PartialShape` - represents dynamic shape. It means that rank or some of dimensions are dynamic (undefined).
`ngraph::PartialShape` can be converted to `ngraph::Shape` using the `get_shape()` method if all dimensions are static; otherwise, conversion raises an exception.

@snippet example_ngraph_utils.cpp ngraph:shape

But in most cases before getting static shape using `get_shape()` method, you need to check that shape is static.  

Also if your transformation requires only input shape rank or particular dimension value, please do not use the `get_shape()` method. See the example below demonstrating how to avoid using `get_shape()`

@snippet example_ngraph_utils.cpp ngraph:shape_check

Not using `get_shape()` method makes your transformation more flexible and applicable for more cases.

###3. Friendly Names

Each `ngraph::Node` has a unique name (used for nGraph internals) and a friendly name. In transformations we care only about friendly name because it represents the name from intermediate representation (IR). 
Also friendly name is used as output tensor name (until we do not have other way to represent output tensor name) and user code that requests intermediate outputs based on these names.
To avoid losing friendly name when replacing node with other node or subgraph, set the original friendly name to the latest node in replacing subgraph. See the example below.

```cpp
// Replace Div operation with Power and Multiply sub-graph and set original friendly name to Multiply operation
auto pow = std::make_shared<ngraph::opset1::Power>(div->input(1).get_source_output(),
                                                           op::Constant::create(div->get_input_element_type(1), Shape{1}, {-1}));
auto mul = std::make_shared<ngraph::opset1::Multiply>(div->input(0).get_source_output(), pow);
mul->set_friendly_name(div->get_friendly_name());
ngraph::replace_node(div, mul);
```

In more advanced cases, when replaced operation has several outputs and we add additional consumers to its outputs, we make a decision how to set friendly name by arrangement.

###4. Runtime Info

Runtime info is a map `std::map<std::string, std::shared_ptr<Variant>>` located inside `ngraph::Node` class. It represents additional attributes in `ngraph::Node`.
These attributes can be set by users or by plugins and when executing transformation that changes `ngraph::Function` we need to preserve these attributes as they will not be automatically propagated.
In most cases, transformations have the following types: 1:1 (replace node with another node), 1:N (replace node with a sub-graph), N:1 (fuse sub-graph into a single node), N:M (any other transformation).
Currently, there is no mechanism that automatically detects transformation types, so we need to propagate this runtime information manually. See the examples below.

```cpp
// Replace Transpose with Reshape operation (1:1)
ngraph::copy_runtime_info(transpose, reshape);
```

```cpp
// Replace Div operation with Power and Multiply sub-graph (1:N)
ngraph::copy_runtime_info(div, {pow, mul});
```

```cpp
// Fuse Convolution with Add operation (N:1)
ngraph::copy_runtime_info({conv, bias}, {conv_ie});
```

```cpp
// Any other transformation that replaces one sub-graph with another sub-graph (N:M)
ngraph::copy_runtime_info({a, b, c}, {e, f});
```

When transformation has multiple fusions or decompositions, `ngraph::copy_runtime_info` must be called multiple times for each case. 

> **Note**: copy_runtime_info removes rt_info from destination nodes. If you want to keep it, you need to specify them in source nodes like this: copy_runtime_info({a, b, c}, {a, b})

###5. Constant Folding

If your transformation inserts constant sub-graphs that need to be folded, do not forget to use `ngraph::pass::ConstantFolding()` after your transformation or call constant folding directly for operation.
The example below shows how constant subgraph can be constructed.

```cpp
// After ConstantFolding pass Power will be replaced with Constant 
auto pow = std::make_shared<ngraph::opset3::Power>(
                    opset3::Constant::create(element::f32, Shape{1}, {2})
                    opset3::Constant::create(element::f32, Shape{1}, {3}));
auto mul = std::make_shared<ngraph::opset3::Multiply>(input /* not constant input */, pow);
``` 

Manual constant folding is more preferable than `ngraph::pass::ConstantFolding()` because it is much faster.

Below you can find an example of manual constant folding:

@snippet src/template_pattern_transformation.cpp manual_constant_folding

## Common mistakes in transformations <a name="common_mistakes"></a>

In transformation development process:

* Do not use deprecated nGraph API. Deprecated methods has the `NGRAPH_DEPRECATED` macros in its definition. 
* Do not pass `shared_ptr<Node>` as an input for other node if type of node is unknown or it has multiple outputs. Use explicit output port.
* If you replace node with another node that produces different shape, remember that new shape will not be propagated until the first `validate_nodes_and_infer_types` call for `ngraph::Function`. If you are using `pass::Manager`, it will automatically call this method after each transformation execution.
* Do not forget to call the `ngraph::ConstantFolding` pass if your transformation creates constant subgraphs.
* Use latest OpSet if you are not developing downgrade transformation pass.
* When developing a callback for `ngraph::pass::MatcherPass`,  do not change nodes that come after the root node in topological order. 

## Using pass manager <a name="using_pass_manager"></a>

`ngraph::pass::Manager` is a container class that can store the list of transformations and execute them. The main idea of this class is to have high-level representation for grouped list of transformations.
It can register and apply any [transformation types](#transformations_types) on function.
In addition, `ngraph::pass::Manager` has extended debug capabilities (find more information in the [how to debug transformations](#how_to_debug_transformations) section). 

The example below shows basic usage of `ngraph::pass::Manager`

@snippet src/template_pattern_transformation.cpp matcher_pass:manager3

Another example shows how multiple matcher passes can be united into single GraphRewrite.

@snippet src/template_pattern_transformation.cpp matcher_pass:manager2

> **Note:** nGraph used to have the `pass::PassConfig` class for transformation pipeline manipulation.
This mechanism is now obsolete and the `pass::PassConfig` class will be removed in future release.

## How to debug transformations <a name="how_to_debug_transformations"></a>

The most popular tool for transformations debugging is the `ngraph::pass::VisualizeTree` transformation, which visualizes ngraph::Function.

Usage example:

@snippet example_ngraph_utils.cpp ngraph:visualize

`ngraph::pass::VisualizeTree` can be parametrized via environment variables:

```
NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES=1 - visualize shapes
NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES=1  - visualize types
```

> **Note**: current VisualTree does not have user-friendly interface and it will be changed in the nearest future. The intention is to move visualization abilities inside transformations.

If you are using `ngraph::pass::Manager` to run sequence of transformations, you can get additional debug capabilities by using the following environment variables:

```
NGRAPH_PROFILE_PASS_ENABLE=1 - enables performance measurement for each transformation and prints execution status
NGRAPH_ENABLE_VISUALIZE_TRACING=1 -  enables visualization after each transformation. By default, it saves dot and svg files.
```

> **Note**: Make sure that you have dot installed on your machine; otherwise, it will silently save only dot file without svg file.

## Disabling/Enabling specific transformations for plugin X	 <a name="disabling_transformation"></a>

In transformation library, we provide plugins transformations like CommonOptimizations, which contains predefined sequence of transformations.
We also provide a tool that helps to disable or partially disable particular transformations in a transformation pipeline.
For example, if a plugin uses the CommonOptimization transformation and needs to disable the ConvertGELU transformation, then inside the plugin we have to take the PassConfig instance
from pass::Manger and call disable method.

@snippet example_ngraph_utils.cpp ngraph:disable_gelu

In some cases, we need to disable transformation for some condition:

@snippet example_ngraph_utils.cpp ngraph:disable_callback

In some cases, pass::Manager pipelines inside transformations may have transformations disabled by default but enabled inside plugins.

@snippet example_ngraph_utils.cpp ngraph:disabled_by_default

PassConfig instance taken from pass::Manager is shared across all registered transformations including nested transformations. So it does not matter where we work with this object (before passes registration or after).

## Transformations testing <a name="transformations_testing"></a>

If you are developing new transformation inside plugin, you need to add test into the `template_plugin/tests/functional/transformations` folder.
We have two types of tests: nGraph reader tests located in `inference-engine/tests/functional/inference_engine/ngraph_reader` and transformation tests located in `inference-engine/tests/functional/inference_engine/transformations`
Reader tests are IR based and test end-to-end conversion from IR to CNNNetwork. Transformation tests test single ngraph transformations or low-level functions that are used inside transformations.

The basic transformation test looks like this:

@snippet tests/functional/transformations/template_transformations_test.cpp transformation:test


[ngraph_replace_node]: ./img/ngraph_replace_node.png
[ngraph_insert_node]: ./img/ngraph_insert_node.png
[transformations_structure]: ./img/transformations_structure.png
[register_new_node]: ./img/register_new_node.png
[graph_rewrite_execution]: ./img/graph_rewrite_execution.png
[graph_rewrite_efficient_search]: ./img/graph_rewrite_efficient_search.png
