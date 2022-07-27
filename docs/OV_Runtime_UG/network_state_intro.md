# Stateful models {#openvino_docs_OV_UG_network_state_intro}

This article describes how to work with stateful networks in OpenVINO™ toolkit. More specifically, it illustrates how stateful networks are represented in IR and nGraph
and how operations with a state can be done. The article additionally provides some examples of stateful networks and code to infer them.

## What is a Stateful Network?

 Several use cases require processing of data sequences. When length of a sequence is known and small enough, 
 it can be processed with RNN like networks that contain a cycle inside. However, in some cases, like online speech recognition of time series 
 forecasting, length of data sequence is unknown. Then, data can be divided in small portions and processed step-by-step. The dependency 
 between data portions should be addressed. For that, networks save some data between inferences - a state. When one dependent sequence is over,
 a state should be reset to initial value and a new sequence can be started.
 
 Several frameworks have special APIs for states in networks. For example, Keras has special option for RNNs, i.e. `stateful` that turns on saving a state 
 between inferences. Kaldi contains special `Offset` specifier to define time offset in a network. 
 
 OpenVINO also contains a special API to simplify work with networks with states. A state is automatically saved between inferences, 
 and there is a way to reset a state when needed. A state can also be read or set to some new value between inferences.
 
## OpenVINO State Representation

 OpenVINO contains the `Variable`, a special abstraction to represent a state in a network. There are two operations to work with a state: 
* `Assign` - to save a value in a state.
* `ReadValue` - to read a value saved on previous iteration.

For more details on these operations, refer to the [ReadValue specification](../ops/infrastructure/ReadValue_3.md) and 
[Assign specification](../ops/infrastructure/Assign_3.md) articles.

## Examples of Networks with States

To get a model with states ready for inference, convert a model from another framework to IR with Model Optimizer or create an nGraph function. (For more information,
refer to the [Build OpenVINO Model section](../OV_Runtime_UG/model_representation.md)). Below is the graph in both forms:

![state_network_example]

### Example of IR with State

The `bin` file for this graph should contain `float 0` in binary form. The content of the `xml` file is as follows.

```xml
<?xml version="1.0" ?>
<net name="summator" version="10">
	<layers>
		<layer id="0" name="init_value" type="Const" version="opset6">
			<data element_type="f32" offset="0" shape="1,1" size="4"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="read" type="ReadValue" version="opset6">
			<data variable_id="id"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="input" type="Parameter" version="opset6">
			<data element_type="f32" shape="1,1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="add_sum" type="Add" version="opset6">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="save" type="Assign" version="opset6">
			<data variable_id="id"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
		</layer>
        <layer id="10" name="add" type="Add" version="opset6">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="output/sink_port_0" type="Result" version="opset6">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
                <edge from-layer="2" from-port="0" to-layer="3" to-port="1"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="0"/>
                <edge from-layer="3" from-port="2" to-layer="4" to-port="0"/>
                <edge from-layer="3" from-port="2" to-layer="10" to-port="0"/> 
                <edge from-layer="1" from-port="1" to-layer="10" to-port="1"/>
                <edge from-layer="10" from-port="2" to-layer="5" to-port="0"/>
	</edges>
	<meta_data>
		<MO_version value="unknown version"/>
		<cli_parameters>
		</cli_parameters>
	</meta_data>
</net>
```

### Example of Creating Model nGraph API

```cpp
	#include <ngraph/opsets/opset6.hpp>
	#include <ngraph/op/util/variable.hpp>
	// ...

    auto arg = make_shared<ngraph::opset6::Parameter>(element::f32, Shape{1, 1});
    auto init_const = ngraph::opset6::Constant::create(element::f32, Shape{1, 1}, {0});

	// The ReadValue/Assign operations must be used in pairs in the network.
	// For each such a pair, its own variable object must be created.
	const std::string variable_name("variable0");
    auto variable = std::make_shared<ngraph::Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name});

	// Creating ngraph::function
    auto read = make_shared<ngraph::opset6::ReadValue>(init_const, variable);
    std::vector<shared_ptr<ngraph::Node>> args = {arg, read};
    auto add = make_shared<ngraph::opset6::Add>(arg, read);
    auto assign = make_shared<ngraph::opset6::Assign>(add, variable);
    auto add2 = make_shared<ngraph::opset6::Add>(add, read);
    auto res = make_shared<ngraph::opset6::Result>(add2);

    auto f = make_shared<Function>(ResultVector({res}), ParameterVector({arg}), SinkVector({assign}));
```

In this example, the `SinkVector` is used to create the `ngraph::Function`. For a network with states, except inputs and outputs, the `Assign` nodes should also point to the `Function` to avoid deleting it during graph transformations. Use the constructor to do it, as shown in the example, or with the special `add_sinks(const SinkVector& sinks)` method. After deleting the node from the graph with the `delete_sink()` method, a sink can be deleted from `ngraph::Function`.

## OpenVINO State API

 Inference Engine has the `InferRequest::QueryState` method to get the list of states from a network and `IVariableState` interface to operate with states. Below is a brief description of methods and the example of how to use this interface.
 
 * `std::string GetName() const` -
   returns the name (variable_id) of a corresponding Variable.
 * `void Reset()` - 
   resets a state to a default value.
 * `void SetState(Blob::Ptr newState)` - 
   sets a new value for a state.
 * `Blob::CPtr GetState() const` - 
   returns current value of state.

## Example of Stateful Network Inference

Based on the IR from the previous section, the example below demonstrates inference of two independent sequences of data. A state should be reset between these sequences.

One infer request and one thread will be used in this example. Using several threads is possible if there are several independent sequences. Then, each sequence can be processed in its own infer request. Inference of one sequence in several infer requests is not recommended. In one infer request, a state will be saved automatically between inferences, but if the first step is done in one infer request and the second in another, a state should be set in a new infer request manually (using the `IVariableState::SetState` method).

@snippet openvino/docs/snippets/InferenceEngine_network_with_state_infer.cpp part1

More elaborate examples demonstrating how to work with networks with states can be found in a speech sample and a demo. 
Refer to the [Samples Overview](./Samples_Overview.md).

[state_network_example]: ./img/state_network_example.png


## LowLatency Transformations

If the original framework does not have a special API for working with states, after importing the model, OpenVINO representation will not contain `Assign`/`ReadValue` layers. For example, if the original ONNX model contains RNN operations, IR will contain `TensorIterator` operations and the values will be obtained only after execution of the whole `TensorIterator` primitive. Intermediate values from each iteration will not be available. Working with these intermediate values of each iteration is enabled by special LowLatency and LowLatency2 transformations, which also help receive these values with a low latency after each infer request.

### How to Get TensorIterator/Loop operations from Different Frameworks via Model Optimizer.

**ONNX and frameworks supported via ONNX format:** `LSTM`, `RNN`, and `GRU` original layers are converted to the `TensorIterator` operation. The `TensorIterator` body contains `LSTM`/`RNN`/`GRU Cell`. The `Peepholes` and `InputForget` modifications are not supported, while the `sequence_lengths` optional input is.
`ONNX Loop` layer is converted to the OpenVINO Loop operation.

**Apache MXNet:** `LSTM`, `RNN`, `GRU` original layers are converted to `TensorIterator` operation. The `TensorIterator` body contains `LSTM`/`RNN`/`GRU Cell` operations.

**TensorFlow:** The `BlockLSTM` is converted to `TensorIterator` operation. The `TensorIterator` body contains `LSTM Cell` operation, whereas `Peepholes` and `InputForget` modifications are not supported.
The `While` layer is converted to `TensorIterator`. The `TensorIterator` body can contain any supported operations. However, when count of iterations cannot be calculated in shape inference (Model Optimizer conversion) time, the dynamic cases are not supported.

**TensorFlow2:** The `While` layer is converted to `Loop` operation. The `Loop` body can contain any supported operations.

**Kaldi:** Kaldi models already contain `Assign`/`ReadValue` (Memory) operations after model conversion. The `TensorIterator`/`Loop` operations are not generated.

## The LowLatencу2 Transformation

The LowLatency2 transformation changes the structure of the network containing [TensorIterator](../ops/infrastructure/TensorIterator_1.md) and [Loop](../ops/infrastructure/Loop_5.md) by adding the ability to work with the state, inserting the `Assign`/`ReadValue` layers as it is shown in the picture below.

### The Differences between the LowLatency and the LowLatency2**:

* Unrolling of `TensorIterator`/`Loop` operations became a part of the LowLatency2, not a separate transformation. After invoking the transformation, the network can be serialized and inferred without re-invoking the transformation.
* Support for `TensorIterator` and `Loop` operations with multiple iterations inside. The `TensorIterator`/`Loop` will not be unrolled in this case.
* The "Parameters connected directly to ReadValues" limitation is resolved. To apply the previous version of the transformation in this case, additional manual manipulations were required. Now, the case is processed automatically.

#### Example of Applying the LowLatency2 Transformation:

<a name="example-of-applying-lowlatency2-transformation"></a>

![applying_low_latency_2_example](./img/applying_low_latency_2.png)

After applying the transformation, the `ReadValue` operations can receive other operations as an input, as shown in the picture above. These inputs should set the initial value for initialization of the `ReadValue` operations. However, such initialization is not supported in the current State API implementation. Input values are ignored and the initial values for the `ReadValue` operations are set to 0 unless otherwise specified by the user via [State API](#openvino-state-api).

### Steps to Apply the LowLatency2 Transformation

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
**Use_const_initializer argument**

By default, the LowLatency2 transformation inserts a constant subgraph of the same shape as the previous input node, and with 0 values as the initializing value for `ReadValue` nodes. (See the picture below.) Insertion of this subgraph can be disabled by passing the `false` value for the `use_const_initializer` argument.

```cpp
InferenceEngine::lowLatency2(cnnNetwork, false);
```

![use_const_initializer_example](./img/llt2_use_const_initializer.png)

**State naming rule:**  A name of a state is a concatenation of names: original `TensorIterator` operation, parameter of the body, and additional suffix `variable_` + `id` (0-base indexing, new indexing for each `TensorIterator`). Use these rules to predict the name of the inserted state after the transformation is applied. For example:

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

4. Use state API. See the [OpenVINO state API](#openvino-state-api) and the [Example of stateful network inference](#example-of-stateful-network-inference) sections.

### Known Limitations
1. Unable to execute the [Reshape](ShapeInference.md) feature to change the number iterations of `TensorIterator`/`Loop` layers to apply the transformation correctly.

	The only way to change the number iterations of `TensorIterator`/`Loop` layer is to use the `Reshape` feature. However, networks can be non-reshapable. The most common reason is that the value of shapes is hardcoded in a constant somewhere in the network. 

	![low_latency_limitation_2](./img/low_latency_limitation_2.png)

	**Current solution:** Trim non-reshapable layers via [ModelOptimizer CLI](../MO_DG/prepare_model/convert_model/Converting_Model.md): the `--input` and `--output` parameters. For example, the parameter and the problematic constant in the picture above can be trimmed using the `--input Reshape_layer_name` command-line option.
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
## [DEPRECATED] The LowLatency Transformation

The LowLatency transformation changes the structure of the network containing [TensorIterator](../ops/infrastructure/TensorIterator_1.md) and [Loop](../ops/infrastructure/Loop_5.md) operations by adding the ability to work with the state, inserting the `Assign`/`ReadValue` layers, as shown in the picture below.

![applying_low_latency_example](./img/applying_low_latency.png)

After applying the transformation, `ReadValue` operations can receive other operations as an input, as shown in the picture above. These inputs should set the initial value for initialization of `ReadValue` operations. However, such initialization is not supported in the current State API implementation. Input values are ignored and the initial values for the `ReadValue` operations are set to 0 unless otherwise specified by the user via [State API](#openvino-state-api).

### Steps to Apply LowLatency Transformation

1. Get CNNNetwork. Either way is acceptable:

	* [from IR or ONNX model](./integrate_with_your_application.md)
	* [from ov::Model](../OV_Runtime_UG/model_representation.md)

2. [Reshape](ShapeInference.md) the CNNNetwork network if necessary. An example of such a **necessary case** is when the `sequence_lengths` dimension of input > 1, 
and it means that `TensorIterator` layer will have `number_iterations` > 1. The inputs of the network should be reshaped to set `sequence_dimension` to exactly 1.

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
**State naming rule:**  a name of a state is a concatenation of names: original `TensorIterator` operation, parameter of the body, and additional suffix `variable_` + `id` (0-base indexing, new indexing for each `TensorIterator`). Use these rules to predict the name of the inserted state after the transformation is applied. For example:

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
4. Use state API. See the [OpenVINO state API](#openvino-state-api) and the [Example of stateful network inference](#example-of-stateful-network-inference) sections.

 
### Known Limitations for the LowLatency [DEPRECATED]
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
