---
sidebar_position: 4
sidebar_label: "Overview of Transformations API"
description: "Learn how to apply additional model optimizations or transform unsupported subgraphs and operations, using OpenVINO™ Transformations API."
---

# Overview of Transformations API


OpenVINO Transformation mechanism allows to develop transformation passes to modify `ov::Model`.
You can use this mechanism to apply additional optimizations to the original Model or transform
unsupported subgraphs and operations to new operations supported by the plugin.
This guide contains all the necessary information to start implementing OpenVINO™ transformations.

## Working with Model

Before moving to the transformation part, it is important to say a few words about the functions which allow modifying `ov::Model`.
This section extends the [model representation guide](../../../openvino-workflow/running-inference/model-representation.md)
and introduces an API for `ov::Model` manipulation.

### Working with node input and output ports

Each OpenVINO operation has `ov::Node` input and output ports, except for `Parameter` and `Constant` types.
The terms `node` and `operation` are used interchangeably in OpenVINO, but this article will maintain consistency in their use.

Every port is associated with a node, allowing access to the node it belongs to, including
its shape, type, all consumers for output ports and the producer node for input ports.

Take a look at the code example:

```cpp
// Let's suppose that node is of ov::op::v0::Convolution type.
// As we know ov::op::v0::Convolution has two input ports (data, weights) and one output port.
ov::Input<ov::Node> data = node->input(0);
ov::Input<ov::Node> weights = node->input(1);
ov::Output<ov::Node> output = node->output(0);

// Getting shape and type
auto pshape = data.get_partial_shape();
auto el_type = data.get_element_type();

// Getting parent for input port i.e. Output mapped by the input
ov::Output<ov::Node> parent_output;
parent_output = data.get_source_output();

// Another short way to get partent for output port
parent_output = node->input_value(0);

// Getting all consumers for output port
auto consumers = output.get_target_inputs();
```

### Node replacement

OpenVINO™ provides two ways for node replacement: via OpenVINO™ helper function and directly
via port methods. We are going to review both of them.

Let's start with OpenVINO™ helper functions. The most popular function is `ov::replace_node(old_node, new_node)`.

Let's review a replacement case where a Negative operation is replaced with Multiply.

![](ov_replace_node.png)

```cpp
bool ov_replace_node(std::shared_ptr<ov::Node> node) {
    // Step 1. Verify that node is of type ov::op::v0::Negative
    auto neg = ov::as_type_ptr<ov::op::v0::Negative>(node);
    if (!neg) {
        return false;
    }

    // Step 2. Create ov::op::v1::Multiply operation with the first input being the output going into Negative and second as Constant with -1 value
    auto mul = std::make_shared<ov::op::v1::Multiply>(neg->input_value(0),
                                                      ov::op::v0::Constant::create(neg->get_element_type(), ov::Shape{1}, {-1}));

    mul->set_friendly_name(neg->get_friendly_name());
    ov::copy_runtime_info(neg, mul);

    // Step 3. Replace Negative operation with Multiply operation
    ov::replace_node(neg, mul);
    return true;

    // Step 4. Negative operation will be removed automatically because all consumers were moved to Multiply operation
}
```

`ov::replace_node` has a constraint that number of output ports for both nodes must be the same.
Otherwise, the attempt to replace the nodes will result in an exception.

The alternative way to do the same replacement is the following:

```cpp
// All neg->output(0) consumers will be moved to mul->output(0) port
neg->output(0).replace(mul->output(0));
```

Another transformation example is insertion. Let's insert an additional Relu node.

![](ov_insert_node.png)

```cpp
// Step 1. Lets suppose that we have a node with single output port and we want to insert additional operation new_node after it
void insert_example(std::shared_ptr<ov::Node> node) {
    // Get all consumers for node
    auto consumers = node->output(0).get_target_inputs();

    // Step 2. Create new node ov::op::v0::Relu.
    auto new_node = std::make_shared<ov::op::v0::Relu>(node);

    // Step 3. Reconnect all consumers to new_node
    for (auto input : consumers) {
        input.replace_source_output(new_node);
    }
}
```

The alternative way of inserting a node is to make a copy of the node and use `ov::replace_node()`:

```cpp
void insert_example_with_copy(std::shared_ptr<ov::Node> node) {
    // Make a node copy
    auto node_copy = node->clone_with_new_inputs(node->input_values());
    // Create new node
    auto new_node = std::make_shared<ov::op::v0::Relu>(node_copy);
    ov::replace_node(node, new_node);
}
```

### Node elimination

Another type of node replacement is elimination of a node.

To eliminate a node, OpenVINO provides a method that considers all limitations of the OpenVINO Runtime.

```cpp
// Suppose we have a node that we want to remove
bool success = ov::replace_output_update_name(node->output(0), node->input_value(0));
```

If the replacement is successful, `ov::replace_output_update_name()` automatically preserves the friendly name and runtime info.


## Transformations types {#transformations-types}

OpenVINO™ Runtime has three main transformation types:

- [Model pass](model-pass.md) - straightforward way to work with `ov::Model` directly
- [Matcher pass](matcher-pass.md) - pattern-based transformation approach
- [Graph rewrite pass](graph-rewrite-pass.md) - container for matcher passes used for efficient execution

![](transformations_structure.png)

## Transformation conditional compilation

Transformation library has two internal macros to support conditional compilation feature.

- `MATCHER_SCOPE(region)` - allows to disable the MatcherPass if matcher isn't used. The region
  name should be unique. This macro creates a local variable `matcher_name` which you should use as a matcher name.
- `RUN_ON_MODEL_SCOPE(region)` - allows to disable run\_on\_model pass if it isn't used. The region name should be unique.


## Transformation writing essentials {#transformation-writing-essentials}

To develop a transformation, follow these transformation rules:

### 1. Friendly Names

Each `ov::Node` has a unique name and a friendly name. In transformations, only the friendly
name matters because it represents the name from the model's perspective.
To prevent losing the friendly name when replacing a node with another node or a subgraph,
the original friendly name is set to the last node in the replacing subgraph. See the example below.

```cpp
// Replace Div operation with Power and Multiply sub-graph and set original friendly name to Multiply operation
auto pow = std::make_shared<ov::op::v1::Power>(div->input(1).get_source_output(),
                                               ov::op::v0::Constant::create(div->get_input_element_type(1), ov::Shape{1}, {-1}));
auto mul = std::make_shared<ov::op::v1::Multiply>(div->input(0).get_source_output(), pow);
mul->set_friendly_name(div->get_friendly_name());
ov::replace_node(div, mul);
```

In more complicated cases, when a replaced operation has several outputs and additional
consumers are added to its outputs, the decision on how to set the friendly name is determined by an agreement.

### 2. Runtime Info

Runtime info is a map `std::map<std::string, ov::Any>` located inside the `ov::Node` class.
It represents additional attributes of the `ov::Node`.
These attributes, which can be set by users or plugins, need to be preserved when executing
a transformation that changes `ov::Model`, as they are not automatically propagated.
In most cases, transformations have the following types: 1:1 (replace node with another node),
1:N (replace node with a sub-graph), N:1 (fuse sub-graph into a single node), N:M (any other transformation).
Currently, there is no mechanism that automatically detects transformation types, so this
runtime information needs to be propagated manually. See the example below:

```cpp
// Replace Transpose with Reshape operation (1:1)
ov::copy_runtime_info(transpose, reshape);

// Replace Div operation with Power and Multiply sub-graph (1:N)
ov::copy_runtime_info(div, {pow, mul});

// Fuse Convolution with Add operation (N:1)
ov::copy_runtime_info({conv, bias}, {conv_fused});

// Any other transformation that replaces one sub-graph with another sub-graph (N:M)
ov::copy_runtime_info({a, b, c}, {e, f});
```

When a transformation has multiple fusions or decompositions, `ov::copy_runtime_info` must be called multiple times for each case.

:::note
`copy_runtime_info` removes `rt_info` from destination nodes. If you want to keep it,
specify them in source nodes as following: `copy_runtime_info({a, b, c}, {a, b})`
:::

### 3. Constant Folding

If your transformation inserts constant sub-graphs that need to be folded, do not forget
to use `ov::pass::ConstantFolding()` after your transformation or call constant folding directly for operation.
The example below shows how constant subgraph can be constructed.

```cpp
// After ConstantFolding pass Power will be replaced with Constant
auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
auto pow = std::make_shared<ov::op::v1::Power>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2}),
                                               ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {3}));
auto mul = std::make_shared<ov::op::v1::Multiply>(input /* not constant input */, pow);
```

Manual constant folding is more preferable than `ov::pass::ConstantFolding()` because it is much faster.

Below you can find an example of manual constant folding:

```cpp
template <class T>
ov::Output<ov::Node> eltwise_fold(const ov::Output<ov::Node>& input0, const ov::Output<ov::Node>& input1) {
    auto eltwise = std::make_shared<T>(input0, input1);
    ov::OutputVector output(eltwise->get_output_size());
    // If constant folding wasn't successful return eltwise output
    if (!eltwise->constant_fold(output, {input0, input1})) {
        return eltwise->output(0);
    }
    return output[0];
}
```


## Common mistakes in transformations {#common-mistakes}

In transformation development process:

- Do not use deprecated OpenVINO™ API. Deprecated methods are marked with `OPENVINO_DEPRECATED` macro in their definition.
- Do not pass `shared_ptr<Node>` as input for another node if the type of the node is unknown
  or if it has multiple outputs. Instead, use explicit output ports.
- If you replace a node with another node that produces different shape, note that
  the new shape will not be propagated until the first `validate_nodes_and_infer_types`
  call for `ov::Model`. If you are using `ov::pass::Manager`, it will automatically call
  this method after each transformation execution.
- Do not forget to call the `ov::pass::ConstantFolding` pass if your transformation creates constant subgraphs.
- Use latest OpSet if you are not developing downgrade transformation pass.
- When developing a callback for `ov::pass::MatcherPass`, do not change nodes that come after the root node in the topological order.


## Using pass manager {#using-pass-manager}

`ov::pass::Manager` is a container class that can store a list of transformations and execute them.
The main idea of this class is to have a high-level representation for grouped list of transformations.
It can register and apply any [transformation pass](#transformations-types) on a model.
In addition, `ov::pass::Manager` has extended debug capabilities (find more information
in the [how to debug transformations](#how-to-debug-transformations) section).

The example below shows basic usage of `ov::pass::Manager`

```cpp
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MyModelTransformation>();
    // Two matchers will run independently (two independent graph traversals)
    // pass::Manager automatically creates GraphRewrite container for each MatcherPass
    manager.register_pass<ov::pass::DecomposeDivideMatcher>();
    manager.register_pass<ov::pass::ReluReluFusionMatcher>();
    manager.run_passes(f);
```

Another example shows how multiple matcher passes can be united into single GraphRewrite.

```cpp
    // Register anchor GraphRewrite pass inside manager that will execute two matchers simultaneously
    ov::pass::Manager manager;
    auto anchor = manager.register_pass<ov::pass::GraphRewrite>();
    using namespace ov::pass;
    ADD_MATCHER(anchor, DecomposeDivideMatcher)
    ADD_MATCHER(anchor, ReluReluFusionMatcher)
    manager.run_passes(f);
```


## How to debug transformations {#how-to-debug-transformations}

If you are using `ov::pass::Manager` to run sequence of transformations, you can get
additional debug capabilities by using the following environment variables:

```cpp
OV_PROFILE_PASS_ENABLE=1 - enables performance measurement for each transformation and prints execution status
OV_ENABLE_VISUALIZE_TRACING=1 -  enables visualization after each transformation. By default, it saves dot and svg files.
```

:::note
Make sure that you have dot installed on your machine; otherwise, it will silently save only dot file without svg file.
:::

## See Also

- [OpenVINO™ Model Representation](../../../openvino-workflow/running-inference/model-representation.md)
- [OpenVINO™ Extensions](../index.md)
