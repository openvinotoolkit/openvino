---
sidebar_position: 4
sidebar_label: "Transformation Patterns with OpenVINO API"
description: "Learn how to apply additional model optimizations or transform unsupported subgraphs and operations using OpenVINO™ Transformations API."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Transformation Patterns with OpenVINO API

Pattern matching is an essential component of OpenVINO™ transformations. Before performing any transformation on a subgraph of a graph, it is necessary to find that subgraph in the graph.
Patterns serve as a searching utility to identify nodes intended for transformations. This article covers the basics of pattern
creation using OpenVINO™ API and helpful utilities to facilitate working with them. While this guide focuses on creating patterns, if you want to learn more about `MatcherPass`, refer to the [OpenVINO Matcher Pass article](matcher-pass.md). Note that some examples may be intentionally simplified for ease of understanding.

Before proceeding further, it is necessary to add some imports. These imports include the operations to be used and additional utilities described in this guide.
Add the following lines to your file:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  import pytest
  from openvino import PartialShape
  from openvino import opset13 as ops
  from openvino.passes import Matcher, WrapType, Or, AnyInput, Optional
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  #include "openvino/op/abs.hpp"
  #include "openvino/op/add.hpp"
  #include "openvino/op/matmul.hpp"
  #include "openvino/op/parameter.hpp"
  #include "openvino/op/relu.hpp"
  #include "openvino/op/sigmoid.hpp"
  #include "openvino/pass/pattern/op/optional.hpp"
  #include "openvino/pass/pattern/op/or.hpp"
  #include "openvino/pass/pattern/op/wrap_type.hpp"
  #include "transformations/utils/utils.hpp"

  using namespace ov;
  using namespace ov::pass;
  using namespace std;
  ```
  </TabItem>
</Tabs>

## Pattern Creation

A pattern is a simplified model comprised of nodes aimed to be matched. It lacks some features of a model and cannot function as one.

Consider a straightforward pattern consisting of three nodes to be found in a given model.

![](simple_pattern_example.png)

Let's create the model and the pattern:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  def simple_model_and_pattern():
      # Create a sample model
      model_param1 = ops.parameter(PartialShape([2, 2]))
      model_param2 = ops.parameter(PartialShape([2, 2]))
      model_add = ops.add(model_param1, model_param2)
      model_param3 = ops.parameter(PartialShape([2, 2]))
      model_mul = ops.matmul(model_add, model_param3, False, False)
      model_abs = ops.abs(model_mul)
      model_relu = ops.relu(model_abs)
      model_result = ops.result(model_relu)

      # Create a sample pattern
      pattern_mul = ops.matmul(AnyInput(), AnyInput(), False, False)
      pattern_abs = ops.abs(pattern_mul)
      pattern_relu = ops.relu(pattern_abs)

      # Create a matcher and try to match the nodes
      matcher = Matcher(pattern_relu, "FindPattern")

      # Should perfectly match
      assert matcher.match(model_relu)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  void create_simple_model_and_pattern() {
      // Create a sample model
      PartialShape shape{2, 2};
      auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
      auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
      auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
      auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
      auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

      // Create a sample model
      auto pattern_mul = std::make_shared<ov::op::v0::MatMul>(pattern::any_input(), pattern::any_input(), false, false);
      auto pattern_abs = std::make_shared<ov::op::v0::Abs>(pattern_mul->output(0));
      auto pattern_relu = std::make_shared<ov::op::v0::Relu>(pattern_abs->output(0));

      // pattern_relu should perfectly match model_relu
  }
  ```
  </TabItem>
</Tabs>

:::note
This example uses testing utilities that directly compare given sequences of nodes. In reality, the process of finding a pattern within a model is more complicated. However, to focus only on patterns and their functionality, these details are intentionally omitted.
:::

Although the code is functional, in OpenVINO, patterns are typically not created using the same nodes as those used for creating the model. Instead, wrappers are preferred, providing additional functionality.
For the given case, `WrapType` is used and the code looks as following:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  def simple_model_and_pattern_wrap_type():
      model_param1 = ops.parameter(PartialShape([2, 2]))
      model_param2 = ops.parameter(PartialShape([2, 2]))
      model_add = ops.add(model_param1, model_param2)
      model_param3 = ops.parameter(PartialShape([2, 2]))
      model_mul = ops.matmul(model_add, model_param3, False, False)
      model_abs = ops.abs(model_mul)
      model_relu = ops.relu(model_abs)
      model_result = ops.result(model_relu)

      # Create a sample pattern
      pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
      pattern_abs = WrapType("opset13.Abs", pattern_mul)
      pattern_relu = WrapType("opset13.Relu", pattern_abs)

      # Create a matcher and try to match the nodes
      matcher = Matcher(pattern_relu, "FindPattern")

      # Should perfectly match
      assert matcher.match(model_relu)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  void create_simple_model_and_pattern_wrap_type() {
      // Create a sample model
      PartialShape shape{2, 2};
      auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
      auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
      auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
      auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
      auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

      // Create a sample model
      auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
      auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
      auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});

      // pattern_relu should perfectly match model_relu
  }
  ```
  </TabItem>
</Tabs>

## 1. WrapType

`WrapType` is a wrapper used to store one or many types to match them. As demonstrated earlier, it is possible to specify a single type in `WrapType` and use it for matching.
However, you can also list all possible types for a given node, for example:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  def wrap_type_list():
      model_param1 = ops.parameter(PartialShape([2, 2]))
      model_param2 = ops.parameter(PartialShape([2, 2]))
      model_add = ops.add(model_param1, model_param2)
      model_param3 = ops.parameter(PartialShape([2, 2]))
      model_mul = ops.matmul(model_add, model_param3, False, False)
      model_abs = ops.abs(model_mul)
      model_relu = ops.relu(model_abs)
      model_result = ops.result(model_relu)
      model_sig = ops.sigmoid(model_abs) # Note that we've added a Sigmoid node after Abs
      model_result1 = ops.result(model_sig)

      # Create a sample pattern
      pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
      pattern_abs = WrapType("opset13.Abs", pattern_mul)
      pattern_relu = WrapType(["opset13.Relu", "opset13.Sigmoid"], pattern_abs)

      # Create a matcher and try to match the nodes
      matcher = Matcher(pattern_relu, "FindPattern")

      # The same pattern perfectly matches 2 different nodes
      assert matcher.match(model_relu)
      assert matcher.match(model_sig)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  void wrap_type_list() {
      // Create a sample model
      PartialShape shape{2, 2};
      auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
      auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
      auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
      auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
      auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));
      auto model_sig = std::make_shared<ov::op::v0::Sigmoid>(model_abs->output(0));
      auto model_result1 = std::make_shared<ov::op::v0::Result>(model_sig->output(0));

      // Create a sample model
      auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
      auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
      auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu, ov::op::v0::Sigmoid>({pattern_abs->output(0)});

      // pattern_relu should perfectly matches model_relu and model_sig
  }
  ```
  </TabItem>
</Tabs>

Note that `pattern_sig` is created with the list `["opset13.Relu", "opset13.Sigmoid"]`, meaning it can be either a `Relu` or a `Sigmoid`.
This feature enables matching the same pattern against different nodes. Essentially, `WrapType` can represent "one of listed" types. `WrapType` supports specifying more than two types.

To add additional checks for your node, create a predicate by providing a function or a lambda. This function will be executed during matching, performing the additional validation specified in the logic of the function. For example, you might want to check the consumers count of a given node:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      WrapType("opset13.Relu", AnyInput(), consumers_count(2))
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
          ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern::any_input()}, pattern::consumers_count(2));
  ```
  </TabItem>
</Tabs>

## 2. AnyInput

`AnyInput` is used when there is no need to specify a particular input for a given node.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      # Create a pattern with a MatMul node taking any inputs.
      pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
      pattern_abs = WrapType("opset13.Abs", pattern_mul)
      pattern_relu = WrapType("opset13.Relu", pattern_abs)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
          auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
          auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
          auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});
  ```
  </TabItem>
</Tabs>

You can also create `AnyInput()` with a predicate, if you want additional checks for you input. It will look similar to `WrapType` with a lambda or a function. For example, to ensure that the input has a rank of 4:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      # Create a pattern with an MatMul node taking any input that has a rank 4.
      pattern_mul = WrapType("opset13.MatMul", [AnyInput(lambda output: len(output.get_shape()) == 4), AnyInput(lambda output: len(output.get_shape()) == 4)])
      pattern_abs = WrapType("opset13.Abs", pattern_mul)
      pattern_relu = WrapType("opset13.Relu", pattern_abs)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
          auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input([](const Output<Node>& value){
                                                                                  return value.get_shape().size() == 4;}),
                                                                              pattern::any_input([](const Output<Node>& value){
                                                                                  return value.get_shape().size() == 4;})});
          auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
          auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});
  ```
  </TabItem>
</Tabs>

## 3. Or

`Or` functions similar to `WrapType`, however, while `WrapType` can only match one of the types provided in the list, `Or` is used to match different branches of nodes.
Suppose the goal is to match the model against two different sequences of nodes. The `Or` type
facilitates this by creating two different branches (`Or` supports more than two branches), looking as follows:

![](or_branches.png)

The red branch will not match, but it will work perfectly for the blue one.
Here is how it looks in code:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  def pattern_or():
      model_param1 = ops.parameter(PartialShape([2, 2]))
      model_param2 = ops.parameter(PartialShape([2, 2]))
      model_add = ops.add(model_param1, model_param2)
      model_param3 = ops.parameter(PartialShape([2, 2]))
      model_mul = ops.matmul(model_add, model_param3, False, False)
      model_abs = ops.abs(model_mul)
      model_relu = ops.relu(model_abs)
      model_result = ops.result(model_relu)

      # Create a red branch
      red_pattern_add = WrapType("opset13.Add", [AnyInput(), AnyInput()])
      red_pattern_relu = WrapType("opset13.Relu", red_pattern_add)
      red_pattern_sigmoid = WrapType(["opset13.Sigmoid"], red_pattern_relu)

      # Create a blue branch
      blue_pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
      blue_pattern_abs = WrapType("opset13.Abs", blue_pattern_mul)
      blue_pattern_relu = WrapType(["opset13.Relu"], blue_pattern_abs)

      #Create Or node
      pattern_or = Or([red_pattern_sigmoid, blue_pattern_relu])

      # Create a matcher and try to match the nodes
      matcher = Matcher(pattern_or, "FindPattern")

      # The same pattern perfectly matches 2 different nodes
      assert matcher.match(model_relu)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  void pattern_or() {
      // Create a sample model
      PartialShape shape{2, 2};
      auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
      auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
      auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
      auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
      auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

      // Create a red branch
      auto red_pattern_add = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
      auto red_pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({red_pattern_add->output(0)});
      auto red_pattern_sigmoid = ov::pass::pattern::wrap_type<ov::op::v0::Sigmoid>({red_pattern_relu->output(0)});

      // Create a blue branch
      auto blue_pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
      auto blue_pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({blue_pattern_mul->output(0)});
      auto blue_pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({blue_pattern_abs->output(0)});

      // Create Or node
      auto pattern_or = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{red_pattern_sigmoid->output(0), blue_pattern_relu->output(0)});

      // pattern_or should perfectly matches model_relu
  }
  ```
  </TabItem>
</Tabs>

Note that matching will succeed for the first matching branch and the remaining ones will not be checked.

## 4. Optional

`Optional` is a bit tricky. It allows specifying whether a node might be present or absent in the model. Under the hood,
the pattern will create two branches using `Or`: one with the optional node present and another one without it. Here is what it would look like with the `Optional`
unfolding into two branches:

![](optional.png)

The code for our model looks as follows:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  def pattern_optional_middle():
      model_param1 = ops.parameter(PartialShape([2, 2]))
      model_param2 = ops.parameter(PartialShape([2, 2]))
      model_add = ops.add(model_param1, model_param2)
      model_param3 = ops.parameter(PartialShape([2, 2]))
      model_mul = ops.matmul(model_add, model_param3, False, False)
      model_abs = ops.abs(model_mul)
      model_relu = ops.relu(model_abs)
      model_result = ops.result(model_relu)

      # Create a sample pattern with an Optional node in the middle
      pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
      pattern_abs = WrapType("opset13.Abs", pattern_mul)
      pattern_sig_opt = Optional(["opset13.Sigmoid"], pattern_abs)
      pattern_relu = WrapType("opset13.Relu", pattern_sig_opt)

      # Create a matcher and try to match the nodes
      matcher = Matcher(pattern_relu, "FindPattern")

      # Should perfectly match
      assert matcher.match(model_relu)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  void pattern_optional_middle() {
      // Create a sample model
      PartialShape shape{2, 2};
      auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
      auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
      auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
      auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
      auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

      // Create a sample pattern with an Optional node in the middle
      auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
      auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
      auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>({pattern_abs->output(0)});
      auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_sig_opt->output(0)});

      // pattern_relu should perfectly match model_relu
  }
  ```
  </TabItem>
</Tabs>

The `Optional` does not necessarily have to be in the middle of the pattern. It can be a top node and a root node.

Top node:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  def pattern_optional_top():
      model_param1 = ops.parameter(PartialShape([2, 2]))
      model_param2 = ops.parameter(PartialShape([2, 2]))
      model_add = ops.add(model_param1, model_param2)
      model_param3 = ops.parameter(PartialShape([2, 2]))
      model_mul = ops.matmul(model_add, model_param3, False, False)
      model_abs = ops.abs(model_mul)
      model_relu = ops.relu(model_abs)
      model_result = ops.result(model_relu)

      # Create a sample pattern an optional top node
      pattern_sig_opt = Optional(["opset13.Sigmoid"], AnyInput())
      pattern_mul = WrapType("opset13.MatMul", [pattern_sig_opt, AnyInput()])
      pattern_abs = WrapType("opset13.Abs", pattern_mul)
      pattern_relu = WrapType("opset13.Relu", pattern_abs)

      matcher = Matcher(pattern_relu, "FindPattern")

      # Should perfectly match even though there's no Sigmoid going into MatMul
      assert matcher.match(model_relu)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  void pattern_optional_top() {
      // Create a sample model
      PartialShape shape{2, 2};
      auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
      auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
      auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
      auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
      auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

      // Create a sample pattern an optional top node
      auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>(pattern::any_input());
      auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern_sig_opt, pattern::any_input()});
      auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
      auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});

      // pattern_relu should perfectly match model_relu
  }
  ```
  </TabItem>
</Tabs>

Root node:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  def pattern_optional_root():
      model_param1 = ops.parameter(PartialShape([2, 2]))
      model_param2 = ops.parameter(PartialShape([2, 2]))
      model_add = ops.add(model_param1, model_param2)
      model_param3 = ops.parameter(PartialShape([2, 2]))
      model_mul = ops.matmul(model_add, model_param3, False, False)
      model_abs = ops.abs(model_mul)
      model_relu = ops.relu(model_abs)
      model_result = ops.result(model_relu)

      # Create a sample pattern
      pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
      pattern_abs = WrapType("opset13.Abs", pattern_mul)
      pattern_relu = WrapType("opset13.Relu", pattern_abs)
      pattern_sig_opt = Optional(["opset13.Sigmoid"], pattern_relu)

      matcher = Matcher(pattern_sig_opt, "FindPattern")

      # Should perfectly match even though there's no Sigmoid as root
      assert matcher.match(model_relu)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  void pattern_optional_root() {
      // Create a sample model
      PartialShape shape{2, 2};
      auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
      auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
      auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
      auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
      auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
      auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

      // Create a sample pattern an optional top node
      auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
      auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
      auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});
      auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>(pattern_relu);

      // pattern_relu should perfectly match model_relu
  }
  ```
  </TabItem>
</Tabs>

`Optional` also supports adding a predicate the same way `WrapType` and `AnyInput` do:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      pattern_sig_opt = Optional(["opset13.Sigmoid"], pattern_relu, consumers_count(1))
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
          auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>(pattern_relu, pattern::consumers_count(2));
  ```
  </TabItem>
</Tabs>
