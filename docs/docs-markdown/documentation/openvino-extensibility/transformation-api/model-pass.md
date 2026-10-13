---
sidebar_position: 1
sidebar_label: "OpenVINO Model Pass"
description: "Learn how to use Model Pass transformation class to take entire ov::Model as input and process it."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenVINO Model Pass

`ov::pass::ModelPass` is used for transformations that take entire `ov::Model` as an input and process it.

Template for ModelPass transformation class

```cpp
// template_model_transformation.hpp
class ov::pass::MyModelTransformation : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MyModelTransformation");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};
```

<Tabs>
  <TabItem value="cpp" label="C++">
  ```cpp
  // template_function_transformation.cpp

  bool ov::pass::MyModelTransformation::run_on_model(const std::shared_ptr<ov::Model>& f) {
      RUN_ON_MODEL_SCOPE(MyModelTransformation);
      // Example transformation code
      NodeVector nodes;

      // Traverse OpenVINO transformation function in topological order
      for (auto& node : f->get_ordered_ops()) {
          // Check that number of input and output ports are equal to 1
          if (node->inputs().size() == 1 && node->outputs().size() == 1) {
              // Check that input and output shape a fully defined (not dynamic) and number of consumers equal to 1
              Input<Node> input = node->input(0);
              Output<Node> output = node->output(0);
              if (input.get_partial_shape().is_static() && output.get_partial_shape().is_static() &&
                  output.get_target_inputs().size() == 1) {
                  nodes.push_back(node);
              }
          }
      }

      // Print types and names for collected nodes
      for (auto& node : nodes) {
          std::cout << "Type: " << node->get_type_info().name << std::endl
                    << "Name: " << node->get_friendly_name() << std::endl;
      }

      // Return false because we didn't change the OpenVINO transformation function
      return false;
  }
  ```
  </TabItem>
  <TabItem value="py" label="Python">
  ```python

  '''
  ``ModelPass`` can be used as a base class for transformation classes that take entire ``Model`` and proceed with it.
  To create transformation, you need to:
  1. Define a class with ``ModelPass`` as a parent.
  2. Redefine the run_on_model method that will receive ``Model`` as an argument.
  '''

  from openvino.passes import ModelPass
  from snippets import get_model

  class MyModelPass(ModelPass):
      def __init__(self):
          super().__init__()

      def run_on_model(self, model):
          for op in model.get_ops():
              print(op.get_friendly_name())


  '''
  This example defines transformation that prints all model operation names.
  The next example shows ModelPass-based transformation usage.
  You create ``Model`` with ``Relu``, ``Parameter`` and ``Result`` nodes. After running this code, you will see the names of the nodes.
  In order to run this script, you need to export PYTHONPATH as the path to binary OpenVINO python models.
  '''

  from openvino.passes import Manager, GraphRewrite, BackwardGraphRewrite, Serialize
  from openvino import opset13 as ops
  from openvino.passes import ModelPass, Matcher, MatcherPass, WrapType


  class MyModelPass(ModelPass):
      def __init__(self):
          super().__init__()

      def run_on_model(self, model):
          for op in model.get_ops():
              print(op.get_friendly_name())


  manager = Manager()
  manager.register_pass(MyModelPass())
  manager.run_passes(get_model())
  ```
  </TabItem>
</Tabs>

Using `ov::pass::ModelPass`, you need to override the `run_on_model` method where you will write the transformation code.
Return value is `true` if the original model has changed during transformation (new operation was added, or operations replacement was made, or node attributes were changed); otherwise, it is `false`.
Also `ov::pass::ModelPass` based transformations can be executed via `ov::pass::Manager`.

## See Also

- [OpenVINO™ Transformations](index.md)
