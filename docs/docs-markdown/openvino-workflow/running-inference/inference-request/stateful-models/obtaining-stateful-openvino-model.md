---
sidebar_position: 1
sidebar_label: "Obtaining a Stateful OpenVINO Model"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Obtaining a Stateful OpenVINO Model

If the original framework does not offer a dedicated API for working with states, the
resulting OpenVINO IR model will not be stateful by default. This means it will not contain
either a state or the [Assign](../../../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/assign-6.md) and
[ReadValue](../../../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/read-value-6.md) operations. You can still
make such models stateful ([see benefits](index.md)),
and you have three ways to do it:

- [Optimum-Intel](https://github.com/huggingface/optimum-intel) - an automated solution
  applicable to a selection of models (not covered by this article, for a usage guide
  refer to the [LLM Inference with Hugging Face and Optimum Intel](../../../../openvino-workflow-generative) article).
- [MakeStateful transformation](#ov-ug-make-stateful) - to choose which pairs of
  Parameter and Result to replace.
- [LowLatency2 transformation](#ov-ug-low-latency) - to detect and replace Parameter
  and Result pairs connected to hidden and cell state inputs of LSTM/RNN/GRU operations
  or Loop/TensorIterator operations.


## MakeStateful Transformation {#ov-ug-make-stateful}

The MakeStateful transformation changes the structure of the model by replacing the
user-defined pairs of Parameter and Results with the Assign and ReadValue operations:

<div style={{textAlign: 'center'}}>

![diagram of MakeStateful Transformation](make_stateful_simple.svg)

</div>

**Only strict syntax is supported**. As shown in the example below, the transformation call
must be enclosed in double quotes "MakeStateful[...]", tensor names - in single quotes
without spaces 'tensor\_name\_1'.

**State naming rule**: in most cases, the name of a state is a concatenation of the
Parameter/Result tensor names. If there are no tensor names,
[friendly names](../../../../documentation/openvino-extensibility/transformation-api/index.md) are used.

**Examples:**

<div style={{textAlign: 'center'}}>

![detailed diagram of MakeStateful Transformation](make_stateful_detailed.png)

</div>

<Tabs>
  <TabItem value="py" label="Python">
  <Tabs>
    <TabItem value="using-tensor-names" label="Using tensor names">
    ```python
        core = ov.Core()
        ov_model = core.read_model("path_to_the_model")
        tensor_names = {"tensor_name_1": "tensor_name_4",
                        "tensor_name_3": "tensor_name_6"}
        manager = Manager()
        manager.register_pass(MakeStateful(tensor_names))
        manager.run_passes(ov_model)
    ```
    </TabItem>
    <TabItem value="using-ops" label="Using Parameter/Result operations">
    ```python
        core = ov.Core()
        ov_model = core.read_model("path_to_the_model")
        # Parameter_1, Result_1, Parameter_3, Result_3 are
        # ops.parameter/ops.result in the ov_model
        pairs = ["""(Parameter_1, Result_1), (Parameter_3, Result_3)"""]
        manager = Manager()
        manager.register_pass(MakeStateful(pairs))
        manager.run_passes(ov_model)
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="cpp" label="C++">
  <Tabs>
    <TabItem value="using-tensor-names" label="Using tensor names">
    ```cpp
        ov::Core core;
        auto ov_model = core.read_model("path_to_the_model");
        std::map<std::string, std::string> tensor_names = {{"tensor_name_1", "tensor_name_4"},
                                                      {"tensor_name_3", "tensor_name_6"}};
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::MakeStateful>(tensor_names);
        manager.run_passes(ov_model);
    ```
    </TabItem>
    <TabItem value="using-ops" label="Using Parameter/Result operations">
    ```cpp
        ov::Core core;
        auto ov_model = core.read_model("path_to_the_model");
        // Parameter_1, Result_1, Parameter_3, Result_3 are shared_ptr<Parameter/Result> in the ov_model
        std::vector<std::pair<std::shared_ptr<ov::opset8::Parameter>, std::shared_ptr<ov::opset8::Result>>> pairs
                = {/*Parameter_1, Result_1, Parameter_3, Result_3*/};
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::MakeStateful>(pairs);
        manager.run_passes(ov_model);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="command-line" label="command line">
  <Tabs>
    <TabItem value="using-tensor-names" label="Using tensor names">
    ```bash
    --input_model <INPUT_MODEL> --transform "MakeStateful[param_res_names={'tensor_name_1':'tensor_name_4','tensor_name_3':'tensor_name_6'}]"
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>


## LowLatency2 Transformation {#ov-ug-low-latency}

The LowLatency2 transformation changes the structure of a model containing
[TensorIterator](../../../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/tensor-iterator-1.md)
and [Loop](../../../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/loop-5.md) by automatically detecting
and replacing pairs of Parameter and Results with the Assign and ReadValue operations,
as illustrated by the following example:

<div style={{textAlign: 'center'}}>

![diagram of LowLatency Transformation](applying_low_latency_2.svg)

</div>

After applying the transformation, ReadValue operations can receive other operations as
input, as shown in the picture above. These inputs should set the initial value for the
initialization of ReadValue operations. However, such initialization is not supported in
the current State API implementation. Input values are ignored, and the initial values
for the ReadValue operations are set to zeros unless the user specifies otherwise via
[State API](index.md).

To apply LowLatency2 Transformation, follow the instruction below:

1. Get [ov::Model](../../model-representation.md),
   for example:
   
   <Tabs>
     <TabItem value="py" label="Python">
     ```python
         core = ov.Core()
         ov_model = core.read_model("path_to_the_model")
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```cpp
         ov::Core core;
         auto ov_model = core.read_model("path_to_the_model");
     ```
     </TabItem>
   </Tabs>
2. Change the number of iterations inside TensorIterator/Loop nodes in the model using the
   [Reshape](../../model-input-output/changing-input-shape.md) feature.
   
   For example, the *sequence\_lengths* dimension of the model input > 1, it means the
   TensorIterator layer has the number\_of\_iterations > 1. You can reshape the model
   inputs to set the *sequence\_dimension* to exactly 1.
   
   <Tabs>
     <TabItem value="py" label="Python">
     ```python
         ov_model.reshape({"X": ov.PartialShape([1, 1, 16])})
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```cpp
         ov_model->reshape({{"X", ov::Shape({1, 1, 16})}});
     ```
     </TabItem>
   </Tabs>
   
   **Unrolling**: If the LowLatency2 transformation is applied to a model containing
   TensorIterator/Loop nodes with exactly one iteration inside, these nodes are unrolled.
   Otherwise, the nodes remain as they are. See the picture above for more details.
3. Apply LowLatency2 transformation.
   
   <Tabs>
     <TabItem value="py" label="Python">
     ```python
         manager = Manager()
         manager.register_pass(LowLatency2())
         manager.run_passes(ov_model)
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```cpp
         ov::pass::Manager manager;
         manager.register_pass<ov::pass::LowLatency2>();
         manager.run_passes(ov_model);
     ```
     </TabItem>
   </Tabs>
   
   (Optional) Use Const Initializer argument:
   
   By default, the LowLatency2 transformation inserts a constant subgraph of the same shape
   as the previous input node. The initializing value for ReadValue nodes is set to zero.
   For more information, see the picture below. You can disable the insertion of this subgraph
   by setting the `use_const_initializer` argument to `false`.
   
   <Tabs>
     <TabItem value="py" label="Python">
     ```python
         manager.register_pass(LowLatency2(False))
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```cpp
         manager.register_pass<ov::pass::LowLatency2>(false);
     ```
     </TabItem>
   </Tabs>
   
   <div style={{textAlign: 'center'}}>

   ![diagram of constant subgraph initialization](llt2_use_const_initializer.svg)

   </div>
   
   **State naming rule:**  the name of a state is a concatenation of several names: the
   original TensorIterator operation, the parameter of the body, and an additional suffix
   `"variable_"` + id (zero-based indexing, new indexing for each TensorIterator). You can
   use these rules to predict the name of the inserted state after applying the transformation.
   For example:
   
   <Tabs>
     <TabItem value="py" label="Python">
     ```python
         # Precondition for Model.
         # TensorIterator and Parameter are created in body of TensorIterator with names
         tensor_iterator_name = "TI_name"
         body_parameter_name = "body_parameter_name"
         idx = "0" # this is a first variable in the model

         # The State will be named "TI_name/param_name/variable_0"
         state_name = tensor_iterator_name + "//" + body_parameter_name + "//" + "variable_" + idx

         #! [ov:get_ov_model]
         core = ov.Core()
         ov_model = core.read_model("path_to_the_model")
         #! [ov:get_ov_model]

         # reshape input if needed

         #! [ov:reshape_ov_model]
         ov_model.reshape({"X": ov.PartialShape([1, 1, 16])})
         #! [ov:reshape_ov_model]

         #! [ov:apply_low_latency_2]
         manager = Manager()
         manager.register_pass(LowLatency2())
         manager.run_passes(ov_model)
         #! [ov:apply_low_latency_2]

         compied_model = core.compile_model(ov_model)
         # Try to find the Variable by name
         infer_request = compied_model.create_infer_request()
         states = infer_request.query_state()
         for state in states:
             name = state.get_name()
             if (name == state_name):
                 # some actions
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```cpp
         // Precondition for ov::Model.
         // TensorIterator and Parameter are created in body of TensorIterator with names
         std::string tensor_iterator_name = "TI_name";
         std::string body_parameter_name = "body_parameter_name";
         std::string idx = "0"; // this is a first variable in the network

         // The State will be named "TI_name/param_name/variable_0"
         auto state_name = tensor_iterator_name + "//" + body_parameter_name + "//" + "variable_" + idx;

         //! [ov:get_ov_model]
         ov::Core core;
         auto ov_model = core.read_model("path_to_the_model");
         //! [ov:get_ov_model]
         // reshape input if needed

         //! [ov:reshape_ov_model]
         ov_model->reshape({{"X", ov::Shape({1, 1, 16})}});
         //! [ov:reshape_ov_model]

         //! [ov:apply_low_latency_2]
         ov::pass::Manager manager;
         manager.register_pass<ov::pass::LowLatency2>();
         manager.run_passes(ov_model);
         //! [ov:apply_low_latency_2]

         auto hd_specific_model = core.compile_model(ov_model);
         // Try to find the Variable by name
         auto infer_request = hd_specific_model.create_infer_request();
         auto states = infer_request.query_state();
         for (auto& state : states) {
             auto name = state.get_name();
             if (name == state_name) {
                 // some actions
             }
         }
     ```
     </TabItem>
   </Tabs>
4. Use state API. See sections [OpenVINO State API](index.md),
   [Stateful Model Inference](index.md#ov-ug-stateful-model-inference).
   
   <div style={{textAlign: 'center'}}>

   ![diagram showing low latency limitation](low_latency_limitation_2.svg)

   </div>
   
   The only way to change the number iterations of TensorIterator/Loop layer is to use the
   [Reshape](../../model-input-output/changing-input-shape.md) feature. However, some models may be
   non-reshapable, typically because the value of shapes is hardcoded in a constant
   somewhere in the model.
   
   In such a case, trim non-reshapable layers via
   [Conversion Parameters](../../../model-preparation/conversion-parameters.md):
   `--input` and `--output`. For example, check the [OpenVINO Model Conversion Tutorial](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/convert-to-openvino).
   
   As for the parameter and the problematic constant in the picture above, it can be
   trimmed by using the `--input Reshape_layer_name` command-line option. The problematic
   constant can be also replaced using OpenVINO, as shown in the following example:
   
   <Tabs>
     <TabItem value="py" label="Python">
     ```python
         # OpenVINO example. How to replace a Constant with hardcoded values of shapes in the model with another one with the new values.
         # Assume we know which Constant (const_with_hardcoded_shape) prevents the reshape from being applied.
         # Then we can find this Constant by name in the model and replace it with a new one with the correct shape.
         core = ov.Core()
         model = core.read_model("path_to_model");
         # Creating the new Constant with a correct shape.
         # For the example shown in the picture above, the new values of the Constant should be 1, 1, 10 instead of 1, 49, 10
         new_const = ops.constant( """value_with_correct_shape, type""")
         for node in model.get_ops():
             # Trying to find the problematic Constant by name.
             if node.get_friendly_name() != "name_of_non_reshapable_const":
                 continue
             # Replacing the problematic Constant with a new one. Do this for all the problematic Constants in the model, then
             # you can apply the reshape feature.
             replace_node(node, new_const)
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```cpp
         // OpenVINO example. How to replace a Constant with hardcoded values of shapes in the network with another one with the new values.
         // Assume we know which Constant (const_with_hardcoded_shape) prevents the reshape from being applied.
         // Then we can find this Constant by name on the network and replace it with a new one with the correct shape.
         ov::Core core;
         auto model = core.read_model("path_to_model");
         // Creating the new Constant with a correct shape.
         // For the example shown in the picture above, the new values of the Constant should be 1, 1, 10 instead of 1, 49, 10
         auto new_const = std::make_shared<ov::opset8::Constant>( /*type, shape, value_with_correct_shape*/ );
         for (const auto& node : model->get_ops()) {
             // Trying to find the problematic Constant by name.
             if (node->get_friendly_name() == "name_of_non_reshapable_const") {
                 auto const_with_hardcoded_shape = ov::as_type_ptr<ov::opset8::Constant>(node);
                 // Replacing the problematic Constant with a new one. Do this for all the problematic Constants in the network, then
                 // you can apply the reshape feature.
                 ov::replace_node(const_with_hardcoded_shape, new_const);
             }
         }
     ```
     </TabItem>
   </Tabs>

## Stateful Model from Scratch

The main approach to obtaining stateful OpenVINO IR models is converting from other
frameworks. Nonetheless, it is possible to create a model from scratch. Check how to
do so in the [Build OpenVINO Model section](../../model-representation.md).

Here is also an example of how `ov::SinkVector` is used to create `ov::Model`. For a
model with states, except inputs and outputs, `Assign` nodes should also point to `Model`
to avoid deleting it during graph transformations. You can do it with the constructor, as in
the example, or with the add\_sinks(const SinkVector& sinks) method. Also, you can delete
a sink from ov::Model after deleting the node from the graph with the delete\_sink() method.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      input = ops.parameter([1, 1], dtype=np.float32, name="data")
      init_const = ops.constant([[0]], dtype=np.float32)

      # Typically ReadValue/Assign operations are presented as pairs in models.
      # ReadValue operation reads information from an internal memory buffer, Assign operation writes data to this buffer.
      # For each pair, its own Variable object must be created.
      # Variable defines name, shape and type of the buffer.
      var_info = VariableInfo()
      var_info.data_shape = init_const.get_shape()
      var_info.data_type = init_const.get_element_type()
      var_info.variable_id = "variable0"
      variable = Variable(var_info)

      # Creating Model
      read = ops.read_value(init_const, variable)
      add = ops.add(input, read)
      assign = ops.assign(add, variable)
      result = ops.result(add)
      model = ov.Model(results=[result], sinks=[assign], parameters=[input], name="model")
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
      // ...

      auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 1});
      auto init_const = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});

      // Typically ReadValue/Assign operations are presented as pairs in models.
      // ReadValue operation reads information from an internal memory buffer, Assign operation writes data to this buffer.
      // For each pair, its own Variable object must be created.
      // Variable defines name, shape and type of the buffer.
      const std::string variable_name("variable0");
      ov::op::util::VariableInfo var_info = {init_const->get_shape(),
                                             init_const->get_element_type(),
                                             variable_name};
      auto variable = std::make_shared<ov::op::util::Variable>(var_info);

      // Creating ov::Model
      auto read = std::make_shared<ov::opset8::ReadValue>(init_const, variable);
      auto add = std::make_shared<ov::opset8::Add>(input, read);
      auto save = std::make_shared<ov::opset8::Assign>(add, variable);
      auto result = std::make_shared<ov::opset8::Result>(add);

      auto model = std::make_shared<ov::Model>(ov::ResultVector({result}),
                                               ov::SinkVector({save}),
                                               ov::ParameterVector({input}));
  ```
  </TabItem>
</Tabs>

:::note
**ONNX and frameworks supported via ONNX format:** *LSTM, RNN, GRU* original layers are
converted to the GRU/RNN/LSTM Sequence operations. *ONNX Loop* layer is converted to the
OpenVINO Loop operation.

**TensorFlow:** *BlockLSTM* is converted to a TensorIterator operation. The TensorIterator
body contains LSTM Cell operation. Modifications such as Peepholes and InputForget are
not supported. The *While* layer is converted to a TensorIterator. The TensorIterator body
can contain any supported operations. However, dynamic cases where the count of iterations
cannot be calculated during shape inference are not supported.

**TensorFlow2:** *While* layer is converted to a Loop operation. The Loop body can contain
any supported operations.
:::
