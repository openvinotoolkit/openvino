# The LowLatencу2 Transformation {#openvino_docs_OV_UG_lowlatency2}

The LowLatency2 transformation changes the structure of the network containing [TensorIterator](../ops/infrastructure/TensorIterator_1.md) and [Loop](../ops/infrastructure/Loop_5.md) by adding the ability to work with the state, inserting the [Assign](../ops/infrastructure/Assign_3.md)/[ReadValue](../ops/infrastructure/ReadValue_3.md) layers as it is shown in the picture below.

## The Differences between the LowLatency and the LowLatency2:

* Unrolling of `TensorIterator`/`Loop` operations became a part of the LowLatency2, not a separate transformation. After invoking the transformation, the network can be serialized and inferred without re-invoking the transformation.
* Support for `TensorIterator` and `Loop` operations with multiple iterations inside. The `TensorIterator`/`Loop` will not be unrolled in this case.
* The "Parameters connected directly to ReadValues" limitation is resolved. To apply the previous version of the transformation in this case, additional manual manipulations were required. Now, the case is processed automatically.

## Example of Applying the Transformation:<a name="example-of-applying-lowlatency2-transformation"></a>

![applying_low_latency_2_example](./img/applying_low_latency_2.png)

After applying the transformation, the `ReadValue` operations can receive other operations as an input, as shown in the picture above. These inputs should set the initial value for initialization of the `ReadValue` operations. However, such initialization is not supported in the current State API implementation. Input values are ignored and the initial values for the `ReadValue` operations are set to 0 unless otherwise specified by the user via [State API](@ref openvino-state-api).

## Steps to Apply LowLatency2

1. Get CNNNetwork. Either way is acceptable:
	* [from IR or ONNX model](./integrate_with_your_application.md)
	* [from ov::Model](../OV_Runtime_UG/model_representation.md)


2. Change the number of iterations inside `TensorIterator`/`Loop` nodes in the network, using the [Reshape](ShapeInference.md) feature. 

   For example, when the `sequence_lengths` dimension of input of the network > 1, the `TensorIterator` layer has `number_iterations` > 1. You can reshape the inputs of the network to set `sequence_dimension` to 1.

   ```cpp

   // Network before reshape: Parameter (name: X, shape: [2 (sequence_lengths), 1, 16]) -> TensorIterator (num_iteration = 2, axis = 0) -> ...

   cnnNetwork.reshape({"X" : {1, 1, 16});

   // Network after reshape: Parameter (name: X, shape: [1 (sequence_lengths), 1, 16]) -> TensorIterator (num_iteration = 1, axis = 0) -> ...
      
   ```
   **Unrolling**: If the LowLatency2 transformation is applied to a network containing `TensorIterator`/`Loop` nodes with exactly one iteration inside, these nodes are unrolled. Otherwise, the nodes remain as they are. For more details, see [the picture](#example-of-applying-lowlatency2-transformation) above.

3. Apply the LowLatency2 transformation.
   ```cpp
   #include "ie_transformations.hpp"

   ...

   InferenceEngine::lowLatency2(cnnNetwork); // 2nd argument 'use_const_initializer = true' by default
   ```
   **Use_const_initializer argument**: By default, the LowLatency2 transformation inserts a constant subgraph of the same shape as the previous input node, and with 0 values as the initializing value for `ReadValue` nodes. (See the picture below.) Insertion of this subgraph can be disabled by passing the `false` value for the `use_const_initializer` argument.

   ```cpp
   InferenceEngine::lowLatency2(cnnNetwork, false);
   ```
   ![use_const_initializer_example](./img/llt2_use_const_initializer.png)

   **State naming rule**: A name of a state is a concatenation of names: original `TensorIterator` operation, parameter of the body, and additional suffix `variable_` + `id` (0-base indexing, new indexing for each `TensorIterator`). Use these rules to predict the name of the inserted state after the transformation is applied. For example:

   ```cpp
      // Precondition in ngraph::function.
      // Created TensorIterator and Parameter in body of TensorIterator with names
      std::string tensor_iterator_name = "TI_name"
      std::string body_parameter_name = "param_name"
      std::string idx = "0"; // it's a first variable in the network
   
      // The State will be named "TI_name/param_name/variable_0"
      auto state_name = tensor_iterator_name + "//" + body_parameter_name + "//" + "variable_" + idx;
   
      InferenceEngine::CNNNetwork cnnNetwork = InferenceEngine::CNNNetwork{function};
      InferenceEngine::lowLatency2(cnnNetwork);
   
      InferenceEngine::ExecutableNetwork executableNetwork = core->LoadNetwork(/*cnnNetwork, targetDevice, configuration*/);
   
      // Try to find the Variable by name
      auto states = executableNetwork.QueryState();
      for (auto& state : states) {
         auto name = state.GetName();
         if (name == state_name) {
            // some actions
         }
      }
   ```


4. Use state API. See the [OpenVINO state API](@ref openvino-state-api) and the [Example of stateful network inference](@ref example-of-stateful-network-inference) sections.

## Known Limitations
1. Unable to execute the [Reshape](ShapeInference.md) feature to change the number iterations of `TensorIterator`/`Loop` layers to apply the transformation correctly.

	The only way to change the number iterations of `TensorIterator`/`Loop` layer is to use the `Reshape` feature. However, networks can be non-reshapable. The most common reason is that the value of shapes is hardcoded in a constant somewhere in the network. 

	![low_latency_limitation_2](./img/low_latency_limitation_2.png)

	**Current solution:** 
   
   * Trim non-reshapable layers via [ModelOptimizer CLI](../MO_DG/prepare_model/convert_model/Converting_Model.md): the `--input` and `--output` parameters. For example, the parameter and the problematic constant in the picture above can be trimmed using the `--input Reshape_layer_name` command-line option.
	The problematic constant can also be replaced using ngraph, as shown in the example below.

   ```cpp
      // nGraph example. How to replace a Constant with hardcoded values of shapes in the network with another one with the new values.
      // Assume we know which Constant (const_with_hardcoded_shape) prevents the reshape from being applied.
      // Then we can find this Constant by name on the network and replace it with a new one with the correct shape.
      auto func = cnnNetwork.getFunction();
      // Creating the new Constant with a correct shape.
      // For the example shown in the picture above, the new values of the Constant should be 1, 1, 10 instead of 1, 49, 10
      auto new_const = std::make_shared<ngraph::opset6::Constant>( /*type, shape, value_with_correct_shape*/ );
      for (const auto& node : func->get_ops()) {
         // Trying to find the problematic Constant by name.
         if (node->get_friendly_name() == "name_of_non_reshapable_const") {
            auto const_with_hardcoded_shape = std::dynamic_pointer_cast<ngraph::opset6::Constant>(node);
            // Replacing the problematic Constant with a new one. Do this for all the problematic Constants in the network, then 
            // you can apply the reshape feature.
            ngraph::replace_node(const_with_hardcoded_shape, new_const);
         }
      }
   ```