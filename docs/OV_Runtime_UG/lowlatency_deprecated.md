# [DEPRECATED] The LowLatency Transformation {#openvino_docs_OV_UG_lowlatency_deprecated}

The deprecated LowLatency transformation changes the structure of the network containing [TensorIterator](../ops/infrastructure/TensorIterator_1.md) and [Loop](../ops/infrastructure/Loop_5.md) operations by adding the ability to work with the state, inserting the [Assign](../ops/infrastructure/Assign_3.md)/[ReadValue](../ops/infrastructure/ReadValue_3.md) layers, as shown in the picture below.

![applying_low_latency_example](./img/applying_low_latency.png)

After applying the transformation, `ReadValue` operations can receive other operations as an input, as shown in the picture above. These inputs should set the initial value for initialization of `ReadValue` operations. However, such initialization is not supported in the current State API implementation. Input values are ignored and the initial values for the `ReadValue` operations are set to 0 unless otherwise specified by the user via [State API](@ref openvino-state-api).

## Steps to Apply LowLatency

1. Get CNNNetwork. Either way is acceptable:

	* [from IR or ONNX model](./integrate_with_your_application.md)
	* [from ov::Model](../OV_Runtime_UG/model_representation.md)

2. [Reshape](ShapeInference.md) the CNNNetwork network if necessary. 

   An example of such a **necessary case** is when the `sequence_lengths` dimension of input > 1, and it means that `TensorIterator` layer will have `number_iterations` > 1. The inputs of the network should be reshaped to set `sequence_dimension` to exactly 1.

   Usually, the following exception, which occurs after applying a transform when trying to infer the network in a plugin, indicates the need to apply the reshape feature: 
   `C++ exception with description "Function is incorrect. The Assign and ReadValue operations must be used in pairs in the network."`
   This means that there are several pairs of `Assign`/`ReadValue` operations with the same `variable_id` in the network and operations were inserted into each iteration of the `TensorIterator`.

   ```cpp

   // Network before reshape: Parameter (name: X, shape: [2 (sequence_lengths), 1, 16]) -> TensorIterator (num_iteration = 2, axis = 0) -> ...

   cnnNetwork.reshape({"X" : {1, 1, 16});

   // Network after reshape: Parameter (name: X, shape: [1 (sequence_lengths), 1, 16]) -> TensorIterator (num_iteration = 1, axis = 0) -> ...
      
   ```

3. Apply the LowLatency transformation.
   ```cpp
   #include "ie_transformations.hpp"

   ...

   InferenceEngine::LowLatency(cnnNetwork);
   ```
   **State naming rule**:  A name of a state is a concatenation of names: original `TensorIterator` operation, parameter of the body, and additional suffix `variable_` + `id` (0-base indexing, new indexing for each `TensorIterator`). Use these rules to predict the name of the inserted state after the transformation is applied. For example:

   ```cpp
      // Precondition in ngraph::function.
      // Created TensorIterator and Parameter in body of TensorIterator with names
      std::string tensor_iterator_name = "TI_name"
      std::string body_parameter_name = "param_name"
      std::string idx = "0"; // it's a first variable in the network

      // The State will be named "TI_name/param_name/variable_0"
      auto state_name = tensor_iterator_name + "//" + body_parameter_name + "//" + "variable_" + idx;

      InferenceEngine::CNNNetwork cnnNetwork = InferenceEngine::CNNNetwork{function};
      InferenceEngine::LowLatency(cnnNetwork);

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

 
## Known Limitations for the LowLatency
1. Parameters connected directly to `ReadValues` (states) after the transformation is applied are not allowed.

	Unnecessary parameters may remain on the graph after applying the transformation. The automatic handling of this case inside the transformation is currently not possible. Such parameters should be removed manually from `ngraph::Function` or replaced with a constant.

	![low_latency_limitation_1](./img/low_latency_limitation_1.png)

	**Current solutions:** 
	* Replace a parameter with a constant (freeze) with the `[0, 0, 0 … 0]` value via [ModelOptimizer CLI](../MO_DG/prepare_model/convert_model/Converting_Model.md): the `--input` or `--freeze_placeholder_with_value` parameters.
	* Use nGraph API to replace a parameter with a constant, as shown in the example below:

		```cpp
		// nGraph example. How to replace Parameter with Constant.
		auto func = cnnNetwork.getFunction();
		// Creating the new Constant with zero values.
		auto new_const = std::make_shared<ngraph::opset6::Constant>( /*type, shape, std::vector with zeros*/ );
		for (const auto& param : func->get_parameters()) {
			// Trying to find the problematic Constant by name.
			if (param->get_friendly_name() == "param_name") {
				// Replacing the problematic Param with a Constant.
				ngraph::replace_node(param, new_const);
				// Removing problematic Parameter from ngraph::function
				func->remove_parameter(param);
			}
		}
		```

2. Unable to execute reshape precondition to apply the transformation correctly.

	Networks can be non-reshapable. The most common reason is that the value of shapes is hardcoded in the constant somewhere in the network. 

	![low_latency_limitation_2](./img/low_latency_limitation_2.png)

	**Current solutions:** 
	* Trim non-reshapable layers via [ModelOptimizer CLI](../MO_DG/prepare_model/convert_model/Converting_Model.md): the `--input` and `--output` parameters. For example, the parameter and the problematic constant (as shown in the picture above) can be trimmed using the `--input Reshape_layer_name` command-line option. 
	* Use nGraph API to replace the problematic constant, as shown in the example below:

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
