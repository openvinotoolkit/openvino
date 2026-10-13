---
sidebar_position: 2
sidebar_label: "OpenVINO Matcher Pass"
description: "Learn how to create a pattern, implement a callback, register the pattern and Matcher to execute MatcherPass transformation on a model."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenVINO Matcher Pass

`ov::pass::MatcherPass`  is used for pattern-based transformations.

Template for MatcherPass transformation class

```cpp
// transformations/template_pattern_transformation.hpp
/**
 * @ingroup ov_transformation_common_api
 * @brief Add transformation description.
 */
class ov::pass::DecomposeDivideMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DecomposeDivideMatcher");
    DecomposeDivideMatcher();
};
```

<Tabs>
  <TabItem value="cpp" label="C++">
  ```cpp
  // template_pattern_transformation.cpp
  ov::pass::DecomposeDivideMatcher::DecomposeDivideMatcher() {
      MATCHER_SCOPE(DecomposeDivideMatcher);
      // Pattern example
      auto input0 = pattern::any_input();
      auto input1 = pattern::any_input();
      auto div = std::make_shared<ov::opset3::Divide>(input0, input1);

      ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
          auto div = ov::as_type_ptr<ov::opset3::Divide>(m.get_match_root());
          // We can not apply this transformation in case with integer input data type
          if (!div || div->input(0).get_element_type().is_integral()) {
              return false;
          }

          // Decompose Divide into Multiply with Power operations
          auto pow = std::make_shared<ov::opset3::Power>(
              div->input_value(1),
              opset3::Constant::create(div->get_input_element_type(1), Shape{1}, {-1}));

          auto mul = std::make_shared<ov::opset3::Multiply>(div->input_value(0), pow);

          // Save original name to last operation in replacement sub-graph
          mul->set_friendly_name(div->get_friendly_name());

          // Copy runtime info attributes to newly created operation
          ov::copy_runtime_info(div, {pow, mul});

          // Replace Divide operation with Multiply
          ov::replace_node(div, mul);

          // Return true as the root node was changed
          return true;
      };

      // Register pattern with Divide operation as a pattern root node
      auto m = std::make_shared<ov::pass::pattern::Matcher>(div, "ConvertDivide");
      // Register Matcher
      register_matcher(m, callback);
  }
  ```
  </TabItem>
  <TabItem value="py" label="Python">
  ```python
  '''
  ``MatcherPass`` is used for pattern-based transformations.
  To create transformation, you need to:

  1. Create a pattern.
  2. Implement a callback.
  3. Register the pattern and ``Matcher``.

  The next example defines transformation that searches for the ``Relu`` layer and inserts after it another
  ``Relu`` layer.
  '''

  from openvino.passes import MatcherPass
  from snippets import get_model

  class PatternReplacement(MatcherPass):
      def __init__(self):
          MatcherPass.__init__(self)
          relu = WrapType("opset13::Relu")

          def callback(matcher: Matcher) -> bool:
              root = matcher.get_match_root()
              new_relu = ops.relu(root.input(0).get_source_output())

              """Use new operation for additional matching
                self.register_new_node(new_relu)

                Input->Relu->Result => Input->Relu->Relu->Result
              """
              root.input(0).replace_source_output(new_relu.output(0))
              return True

          self.register_matcher(Matcher(relu, "PatternReplacement"), callback)


  '''
  After running this code, you will see the next:

  model ops :
  parameter
  result
  relu

  model ops :
  parameter
  result
  relu
  new_relu

  In order to run this script, you need to export PYTHONPATH as the path to binary OpenVINO python models.
  '''
  from openvino.passes import Manager
  from openvino import opset13 as ops
  from openvino.passes import Matcher, MatcherPass, WrapType

  class PatternReplacement(MatcherPass):
      def __init__(self):
          MatcherPass.__init__(self)
          relu = WrapType("opset13::Relu")

          def callback(matcher: Matcher) -> bool:
              root = matcher.get_match_root()
              new_relu = ops.relu(root.input(0).get_source_output())
              new_relu.set_friendly_name('new_relu')

              """Use new operation for additional matching
                self.register_new_node(new_relu)

                Input->Relu->Result => Input->Relu->Relu->Result
              """
              root.input(0).replace_source_output(new_relu.output(0))
              return True

          self.register_matcher(Matcher(relu, "PatternReplacement"), callback)


  def print_model_ops(model):
      print('model ops : ')
      for op in model.get_ops():
          print(op.get_friendly_name())
      print('')


  manager = Manager()
  manager.register_pass(PatternReplacement())


  model = get_model()
  print_model_ops(model)
  manager.run_passes(model)
  print_model_ops(model)
  ```
  </TabItem>
</Tabs>

To use `ov::pass::MatcherPass`, you need to complete these steps:

1. Create a pattern
2. Implement a callback
3. Register the pattern and Matcher
4. Execute MatcherPass

So let's go through each of these steps.

## Create a pattern

Pattern is a single root `ov::Model`. But the only difference is that you do not need to create a model object, you just need to create and connect opset or special pattern operations.
Then you need to take the last created operation and put it as a root of the pattern. This root node will be used as a root node in pattern matching.

:::note
Any nodes in a pattern that have no consumers and are not registered as root will not be used in pattern matching.
:::

```cpp
// Pattern example
auto input = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::Shape{1});
auto shapeof = std::make_shared<ov::opset8::ShapeOf>(input);

// Create Matcher with Parameter->ShapeOf pattern
auto m = std::make_shared<ov::pass::pattern::Matcher>(shapeof, "MyPatternBasedTransformation");
```

The `Parameter` operation in the example above has type and shape specified. These attributes are needed only to create Parameter operation class and will not be used in pattern matching.

For more pattern examples, refer to the [pattern matching section](#pattern-matching).

## Implement callback

Callback is an action applied to every pattern entrance. In general, callback is the lambda function that takes Matcher object with detected subgraph.

```cpp
ov::graph_rewrite_callback callback = [](ov::pass::pattern::Matcher& m) {
    // Get root node
    std::shared_ptr<ov::Node> root_node = m.get_match_root();

    // Get all nodes matched by pattern
    ov::NodeVector nodes = m.get_matched_nodes();

    // Transformation code
    return false;
};
```

The example above shows the callback structure and how Matcher can be used for accessing nodes detected by pattern.
Callback return value is `true` if root node was replaced and another pattern cannot be applied to the same root node; otherwise, it is `false`.

:::note
It is not recommended to manipulate with nodes that are under root node. This may affect GraphRewrite execution as it is expected that all nodes that come after root node in topological order are valid and can be used in pattern matching.
:::

MatcherPass also provides functionality that allows reporting of the newly created nodes that can be used in additional pattern matching.
If MatcherPass was registered in `ov::pass::Manager` or `ov::pass::GraphRewrite`, these registered nodes will be added for additional pattern matching.
That means that matcher passes registered in `ov::pass::GraphRewrite` will be applied to these nodes.

The example below shows how single MatcherPass can fuse sequence of operations using the `register_new_node` method.

```cpp
ov::pass::ReluReluFusionMatcher::ReluReluFusionMatcher() {
    MATCHER_SCOPE(ReluReluFusionMatcher);
    auto m_relu1 = ov::pass::pattern::wrap_type<ov::opset3::Relu>(pattern::consumers_count(1));
    auto m_relu2 = ov::pass::pattern::wrap_type<ov::opset3::Relu>({m_relu1});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        // Map that helps to connect labels with matched outputs
        auto& node_to_output = m.get_pattern_value_map();

        // Create new Relu operation and add register it for additional execution
        auto new_relu =
            register_new_node<ov::opset3::Relu>(node_to_output.at(m_relu1).get_node_shared_ptr()->input_value(0));

        // Copy runtime info attributes to newly created operation
        ov::copy_runtime_info(m.get_matched_nodes(), new_relu);

        // Save last Relu name to new Relu operation
        new_relu->set_friendly_name(m.get_match_root()->get_friendly_name());

        // Replace Relu->Relu with Relu
        ov::replace_node(m.get_match_root(), new_relu);

        // Return true as the root node was changed
        return true;
    };

    // Register pattern with Relu operation as a pattern root node
    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_relu2, "ReluReluFusion");
    // Register Matcher
    register_matcher(m, callback);
}
```

:::note
If you register multiple nodes, please add them in topological order. We do not topologically sort these nodes as it is a time-consuming operation.
:::

## Register pattern and Matcher

The last step is to register Matcher and callback inside the MatcherPass pass. To do this, call the `register_matcher` method.

:::note
Only one matcher can be registered for a single MatcherPass class.
:::

```cpp
// Register matcher and callback
register_matcher(m, callback);
```

## Execute MatcherPass

MatcherPass has multiple ways to be executed:

- Run on a single node - it can be useful if you want to run MatcherPass inside another transformation.

```cpp
    if (ov::pass::DecomposeDivideMatcher().apply(node)) {
        // successful execution (root node was replaced)
    }
```

- Run on `ov::Model` using GraphRewrite - this approach gives ability to run MatcherPass on whole `ov::Model`. Moreover, multiple MatcherPass transformation can be registered in a single GraphRewite to be executed in a single graph traversal.

```cpp
    // Two matcher passes will run simultaneously in a single graph traversal
    ov::pass::GraphRewrite pass;
    pass.add_matcher<ov::pass::DecomposeDivideMatcher>();
    pass.add_matcher<ov::pass::ReluReluFusionMatcher>();
    pass.run_on_model(f);
```

- Run on `ov::Model` using `ov::pass::Manager` - this approach helps you to register MatcherPass for execution on `ov::Model` as another transformation types.

```cpp
    // Two matchers will run independently (two independent graph traversals)
    // pass::Manager automatically creates GraphRewrite container for each MatcherPass
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::DecomposeDivideMatcher>();
    manager.register_pass<ov::pass::ReluReluFusionMatcher>();
    manager.run_passes(f);
```

## Pattern Matching

Sometimes patterns cannot be expressed via regular operations or it is too complicated.
For example, if you want to detect **Convolution->Add** sub-graph without specifying particular input type for Convolution operation or you want to create a pattern where some of operations can have different types.
And for these cases OpenVINO™ provides additional helpers to construct patterns for GraphRewrite transformations.

There are two main helpers:

1. `ov::pass::pattern::any_input` - helps to express inputs if their types are undefined.
2. `ov::pass::pattern::wrap_type <T>` - helps to express nodes of pattern without specifying node attributes.

Let's go through the example to have better understanding of how it works:

:::note
Node attributes do not participate in pattern matching and are needed only for operations creation. Only operation types participate in pattern matching.
:::

The example below shows basic usage of `ov::passpattern::any_input`.
Here we construct Multiply pattern with arbitrary first input and Constant as a second input.
Also as Multiply is commutative operation, it does not matter in which order we set inputs (any\_input/Constant or Constant/any\_input) because both cases will be matched.

```cpp
// Detect Multiply with arbitrary first input and second as Constant
// ov::pattern::op::Label - represent arbitrary input
auto input = ov::pass::pattern::any_input();
auto value = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {0.5});
auto mul = std::make_shared<ov::opset8::Multiply>(input, value);
auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, "MultiplyMatcher");
```

This example shows how we can construct a pattern when operation has arbitrary number of inputs.

```cpp
// Detect Concat operation with arbitrary number of inputs
auto concat = ov::pass::pattern::wrap_type<ov::opset8::Concat>();
auto m = std::make_shared<ov::pass::pattern::Matcher>(concat, "ConcatMatcher");
```

This example shows how to use predicate to construct a pattern. Also it shows how to match pattern manually on given node.

```cpp
// Detect Multiply->Add sequence where mul has exactly one consumer
auto mul = ov::pass::pattern::wrap_type<ov::opset8::Multiply>(ov::pass::pattern::consumers_count(1)/*сheck consumers count*/);
auto add = ov::pass::pattern::wrap_type<ov::opset8::Add>({mul, ov::pass::pattern::any_input()});
auto m = std::make_shared<ov::pass::pattern::Matcher>(add, "MultiplyAddMatcher");
// Matcher can be used to match pattern manually on given node
if (m->match(node->output(0))) {
    // Successfully matched
}
```

:::note
Be careful with manual matching because Matcher object holds matched nodes. To clear a match, use the m->clear\_state() method.
:::

## See Also

- [OpenVINO™ Transformations](index.md)
