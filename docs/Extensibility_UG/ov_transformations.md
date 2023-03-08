# Overview of Transformations API {#openvino_docs_transformations}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_Extensibility_UG_model_pass
   openvino_docs_Extensibility_UG_matcher_pass
   openvino_docs_Extensibility_UG_graph_rewrite_pass

@endsphinxdirective

OpenVINO Transformation mechanism allows to develop transformation passes to modify `ov::Model`. You can use this mechanism to apply additional optimizations to the original Model or transform unsupported subgraphs and operations to new operations which are supported by the plugin.
This guide contains all necessary information that you need to start implementing OpenVINO™ transformations.

## Working with Model

Before the moving to transformation part it is needed to say several words about functions which allow to modify `ov::Model`.
This chapter extends the [model representation guide](../OV_Runtime_UG/model_representation.md) and shows an API that allows us to manipulate with `ov::Model`.

### Working with node input and output ports

First of all let's talk about `ov::Node` input/output ports. Each OpenVINO™ operation has input and output ports except cases when operation has `Parameter` or `Constant` type.

Every port belongs to its node, so using a port we can access parent node, get shape and type for particular input/output, get all consumers in case of output port, and get producer node in case of input port.
With output port we can set inputs for newly created operations.

Lets look at the code example.

@snippet ov_model_snippets.cpp ov:ports_example

### Node replacement

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

### Node elimination

Another type of node replacement is its elimination.

To eliminate operation, OpenVINO™ has special method that considers all limitations related to OpenVINO™ Runtime.

@snippet ov_model_snippets.cpp ov:eliminate_node

`ov::replace_output_update_name()` in case of successful replacement it automatically preserves friendly name and runtime info.

## Transformations types <a name="transformations-types"></a>

OpenVINO™ Runtime has three main transformation types:

* [Model pass](./model_pass.md) - straightforward way to work with `ov::Model` directly
* [Matcher pass](./matcher_pass.md) - pattern-based transformation approach
* [Graph rewrite pass](./graph_rewrite_pass.md) - container for matcher passes needed for efficient execution

![transformations_structure]

## Transformation conditional compilation

Transformation library has two internal macros to support conditional compilation feature.

* `MATCHER_SCOPE(region)` - allows to disable the MatcherPass if matcher isn't used. The region name should be unique. This macro creates a local variable `matcher_name` which you should use as a matcher name.
* `RUN_ON_MODEL_SCOPE(region)` - allows to disable run_on_model pass if it isn't used. The region name should be unique.

## Transformation writing essentials <a name="transformation_writing_essentials"></a>

When developing a transformation, you need to follow these transformation rules:

### 1. Friendly Names

Each `ov::Node` has an unique name and a friendly name. In transformations we care only about friendly name because it represents the name from the model.
To avoid losing friendly name when replacing node with other node or subgraph, set the original friendly name to the latest node in replacing subgraph. See the example below.

@snippet ov_model_snippets.cpp ov:replace_friendly_name

In more advanced cases, when replaced operation has several outputs and we add additional consumers to its outputs, we make a decision how to set friendly name by arrangement.

### 2. Runtime Info

Runtime info is a map `std::map<std::string, ov::Any>` located inside `ov::Node` class. It represents additional attributes in `ov::Node`.
These attributes can be set by users or by plugins and when executing transformation that changes `ov::Model` we need to preserve these attributes as they will not be automatically propagated.
In most cases, transformations have the following types: 1:1 (replace node with another node), 1:N (replace node with a sub-graph), N:1 (fuse sub-graph into a single node), N:M (any other transformation).
Currently, there is no mechanism that automatically detects transformation types, so we need to propagate this runtime information manually. See the examples below.

@snippet ov_model_snippets.cpp ov:copy_runtime_info

When transformation has multiple fusions or decompositions, `ov::copy_runtime_info` must be called multiple times for each case.

> **NOTE**: `copy_runtime_info` removes `rt_info` from destination nodes. If you want to keep it, you need to specify them in source nodes like this: `copy_runtime_info({a, b, c}, {a, b})`

### 3. Constant Folding

If your transformation inserts constant sub-graphs that need to be folded, do not forget to use `ov::pass::ConstantFolding()` after your transformation or call constant folding directly for operation.
The example below shows how constant subgraph can be constructed.

@snippet ov_model_snippets.cpp ov:constant_subgraph

Manual constant folding is more preferable than `ov::pass::ConstantFolding()` because it is much faster.

Below you can find an example of manual constant folding:

@snippet template_pattern_transformation.cpp manual_constant_folding

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
It can register and apply any [transformation pass](#transformations-types) on model.
In addition, `ov::pass::Manager` has extended debug capabilities (find more information in the [how to debug transformations](#how-to-debug-transformations) section).

The example below shows basic usage of `ov::pass::Manager`

@snippet template_pattern_transformation.cpp matcher_pass:manager3

Another example shows how multiple matcher passes can be united into single GraphRewrite.

@snippet template_pattern_transformation.cpp matcher_pass:manager2

## How to debug transformations <a name="how-to-debug-transformations"></a>

If you are using `ngraph::pass::Manager` to run sequence of transformations, you can get additional debug capabilities by using the following environment variables:

```
OV_PROFILE_PASS_ENABLE=1 - enables performance measurement for each transformation and prints execution status
OV_ENABLE_VISUALIZE_TRACING=1 -  enables visualization after each transformation. By default, it saves dot and svg files.
```

> **NOTE**: Make sure that you have dot installed on your machine; otherwise, it will silently save only dot file without svg file.

## See Also

* [OpenVINO™ Model Representation](../OV_Runtime_UG/model_representation.md)
* [OpenVINO™ Extensions](./Intro.md)

[ngraph_replace_node]: ./img/ngraph_replace_node.png
[ngraph_insert_node]: ./img/ngraph_insert_node.png
[transformations_structure]: ./img/transformations_structure.png
[register_new_node]: ./img/register_new_node.png
