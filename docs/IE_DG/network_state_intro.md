Introduction to OpenVINO state API {#openvino_docs_IE_DG_network_state_intro}
==============================

This section describes how to work with stateful networks in OpenVINO toolkit, specifically:
* How stateful networks are represented in IR and nGraph
* How operations with state can be done

The section additionally provides small examples of stateful network and code to infer it.

## What is a Stateful Network

 Several use cases require processing of data sequences. When length of a sequence is known and small enough, 
 we can process it with RNN like networks that contain a cycle inside. But in some cases, like online speech recognition of time series 
 forecasting, length of data sequence is unknown. Then data can be divided in small portions and processed step-by-step. But dependency 
 between data portions should be addressed. For that, networks save some data between inferences - state. When one dependent sequence is over,
 state should be reset to initial value and new sequence can be started.
 
 Several frameworks have special API for states in networks. For example, Keras have special option for RNNs `stateful` that turns on saving state 
 between inferences. Kaldi contains special specifier `Offset` to define time offset in a network. 
 
 OpenVINO also contains special API to simplify work with networks with states. State is automatically saved between inferences, 
 and there is a way to reset state when needed. You can also read state or set it to some new value between inferences.
 
## OpenVINO State Representation

 OpenVINO contains a special abstraction `Variable` to represent a state in a network. There are two operations to work with the state: 
* `Assign` to save value in state
* `ReadValue` to read value saved on previous iteration

You can find more details on these operations in [ReadValue specification](../ops/infrastructure/ReadValue_3.md) and 
[Assign specification](../ops/infrastructure/Assign_3.md).

## Examples of Representation of a Network with States

To get a model with states ready for inference, you can convert a model from another framework to IR with Model Optimizer or create an nGraph function (details can be found in [Build nGraph Function section](../nGraph_DG/build_function.md)). Let's represent the following graph in both forms:

![state_network_example]

### Example of IR with State

The `bin` file for this graph should contain float 0 in binary form. Content of `xml` is the following.

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

In this example, `SinkVector` is used to create `ngraph::Function`. For network with states, except inputs and outputs,  `Assign` nodes should also point to `Function` 
to avoid deleting it during graph transformations. You can do it with the constructor, as shown in the example, or with the special method `add_sinks(const SinkVector& sinks)`. Also you can delete 
sink from `ngraph::Function` after deleting the node from graph with the `delete_sink()` method.

## OpenVINO state API

 Inference Engine has the `InferRequest::QueryState` method  to get the list of states from a network and `IVariableState` interface to operate with states. Below you can find brief description of methods and the workable example of how to use this interface.
 
 * `std::string GetName() const`
   returns name(variable_id) of according Variable
 * `void Reset()`
   reset state to default value
 * `void SetState(Blob::Ptr newState)`
   set new value for state
 * `Blob::CPtr GetState() const`
   returns current value of state

## Example of Stateful Network Inference

Let's take an IR from the previous section example. The example below demonstrates inference of two independent sequences of data. State should be reset between these sequences.

One infer request and one thread 
will be used in this example. Using several threads is possible if you have several independent sequences. Then each sequence can be processed in its own infer 
request. Inference of one sequence in several infer requests is not recommended. In one infer request state will be saved automatically between inferences, but 
if the first step is done in one infer request and the second in another, state should be set in new infer request manually (using `IVariableState::SetState` method).

@snippet openvino/docs/snippets/InferenceEngine_network_with_state_infer.cpp part1

You can find more powerful examples demonstrating how to work with networks with states in speech sample and demo. 
Decsriptions can be found in [Samples Overview](./Samples_Overview.md)

[state_network_example]: ./img/state_network_example.png


## LowLatency Transformation

If the original framework does not have a special API for working with states, after importing the model, OpenVINO representation will not contain Assign/ReadValue layers. For example, if the original ONNX model contains RNN operations, IR will contain TensorIterator operations and the values will be obtained only after the execution of whole TensorIterator primitive, intermediate values from each iteration will not be available. To be able to work with these intermediate values of each iteration and receive them with a low latency after each infer request, a special LowLatency transformation was introduced.

LowLatency transformation changes the structure of the network containing [TensorIterator](../ops/infrastructure/TensorIterator_1.md) and [Loop](../ops/infrastructure/Loop_5.md) by adding the ability to work with the state, inserting the Assign/ReadValue layers as it is shown in the picture below.

![applying_low_latency_example](./img/applying_low_latency.png)

After applying the transformation, ReadValue operations can receive other operations as an input, as shown in the picture above. These inputs should set the initial value for initialization of ReadValue operations. However, such initialization is not supported in the current State API implementation. Input values are ignored and the initial values for the ReadValue operations are set to zeros unless otherwise specified by the user via [State API](#openvino-state-api).

### Steps to apply LowLatency Transformation

1. Get CNNNetwork. Either way is acceptable:

	* [from IR or ONNX model](./Integrate_with_customer_application_new_API.md)
	* [from nGraph Function](../nGraph_DG/build_function.md)

2. [Reshape](ShapeInference.md) the CNNNetwork network if necessary. **Necessary case:** where the sequence_lengths dimension of input > 1, it means TensorIterator layer will have number_iterations > 1. We should reshape the inputs of the network to set sequence_dimension to exactly 1.

Usually, the following exception, which occurs after applying a transform when trying to infer the network in a plugin, indicates the need to apply reshape feature: `C++ exception with description "Function is incorrect. Assign and ReadValue operations must be used in pairs in the network."`
This means that there are several pairs of Assign/ReadValue operations with the same variable_id in the network, operations were inserted into each iteration of the TensorIterator.

```cpp

// Network before reshape: Parameter (name: X, shape: [2 (sequence_lengths), 1, 16]) -> TensorIterator (num_iteration = 2, axis = 0) -> ...

cnnNetwork.reshape({"X" : {1, 1, 16});

// Network after reshape: Parameter (name: X, shape: [1 (sequence_lengths), 1, 16]) -> TensorIterator (num_iteration = 1, axis = 0) -> ...
	
```

3. Apply LowLatency transformation
```cpp
#include "ie_transformations.hpp"

...

InferenceEngine::LowLatency(cnnNetwork);
```
**State naming rule:**  a name of a state is a concatenation of names: original TensorIterator operation, Parameter of the body, and additional suffix "variable_" + id (0-base indexing, new indexing for each TensorIterator). You can use these rules to predict what the name of the inserted State will be after the transformation is applied. For example:
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
4. Use state API. See sections [OpenVINO state API](#openvino-state-api), [Example of stateful network inference](#example-of-stateful-network-inference).

 
### Known Limitations
1. Parameters connected directly to ReadValues (States) after the transformation is applied are not allowed.

	Unnecessary parameters may remain on the graph after applying the transformation. The automatic handling of this case inside the transformation is not possible now. Such Parameters should be removed manually from `ngraph::Function` or replaced with a Constant.

	![low_latency_limitation_1](./img/low_latency_limitation_1.png)

	**Current solutions:** 
	* Replace Parameter with Constant (freeze) with the value [0, 0, 0 … 0] via [ModelOptimizer CLI](../MO_DG/prepare_model/convert_model/Converting_Model_General.md) `--input` or `--freeze_placeholder_with_value`.
	* Use ngraph API to replace Parameter with Constant.

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

2.  Unable to execute reshape precondition to apply the transformation correctly due to hardcoded values of shapes somewhere in the network.

	Networks can be non-reshapable, the most common reason is that the value of shapes is hardcoded in the Constant somewhere in the network. 

	![low_latency_limitation_2](./img/low_latency_limitation_2.png)

	**Current solution:** trim non-reshapable layers via [ModelOptimizer CLI](../MO_DG/prepare_model/convert_model/Converting_Model_General.md) `--input`, `--output`. For example, we can trim the Parameter and the problematic Constant in the picture above, using the following command line option: 
	`--input Reshape_layer_name`. We can also replace the problematic Constant using ngraph, as shown in the example below.

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
