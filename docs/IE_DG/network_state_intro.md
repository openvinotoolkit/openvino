Introduction to OpenVINO state API {#openvino_docs_IE_DG_network_state_intro}
==============================

This section describes how to work with stateful networks in OpenVINO toolkit, specifically:
* How stateful networks are represented in IR and nGraph
* How operations with state can be done

The section additionally provides small examples of stateful network and code to infer it.

## What is a stateful network

 Several use cases require processing of data sequences. When length of a sequence is known and small enough, 
 we can process it with RNN like networks that contain a cycle inside. But in some cases, like online speech recognition of time series 
 forecasting, length of data sequence is unknown. Then data can be divided in small portions and processed step-by-step. But dependency 
 between data portions should be addressed. For that, networks save some data between inferences - state. When one dependent sequence is over,
 state should be reset to initial value and new sequence can be started.
 
 Several frameworks have special API for states in networks. For example, Keras have special option for RNNs `stateful` that turns on saving state 
 between inferences. Kaldi contains special specifier `Offset` to define time offset in a network. 
 
 OpenVINO also contains special API to simplify work with networks with states. State is automatically saved between inferences, 
 and there is a way to reset state when needed. You can also read state or set it to some new value between inferences.
 
## OpenVINO state representation

 OpenVINO contains special abstraction variable to represent state in a network. There are two operations to work with state: 
* `Assign` to save value in state
* `ReadValue` to read value saved on previous iteration

You can find more details on these operations in [ReadValue specification](../ops/infrastructure/ReadValue_3.md) and 
[Assign specification](../ops/infrastructure/Assign_3.md).

## Examples of representation of a network with states

To get a model with states ready for inference, you can convert a model from another framework to IR with Model Optimizer or create an nGraph function 
(details can be found in [Build nGraph Function section](../nGraph_DG/build_function.md)). 
Let's represent the following graph in both forms:
![state_network_example]

### Example of IR with state

The `bin` file for this graph should contain float 0 in binary form. Content of `xml` is the following.

```xml
<?xml version="1.0" ?>
<net name="summator" version="10">
	<layers>
		<layer id="0" name="init_value" type="Const" version="opset5">
			<data element_type="f32" offset="0" shape="1,1" size="4"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="read" type="ReadValue" version="opset5">
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
		<layer id="2" name="input" type="Parameter" version="opset5">
			<data element_type="f32" shape="1,1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="add_sum" type="Add" version="opset5">
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
		<layer id="4" name="save" type="Assign" version="opset5">
			<data variable_id="id"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
		</layer>
                <layer id="10" name="add" type="Add" version="opset5">
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
		<layer id="5" name="output/sink_port_0" type="Result" version="opset5">
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

### Example of creating model nGraph API

```cpp
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 1});
    auto init_const = op::Constant::create(element::f32, Shape{1, 1}, {0});
    auto read = make_shared<op::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto add = make_shared<op::Add>(arg, read);
    auto assign = make_shared<op::Assign>(add, "v0");
    auto add2 = make_shared<op::Add>(add, read);
    auto res = make_shared<op::Result>(add2);

    auto f = make_shared<Function>(ResultVector({res}), ParameterVector({arg}), SinkVector({assign}));
```

In this example, `SinkVector` is used to create `ngraph::Function`. For network with states, except inputs and outputs,  `Assign` nodes should also point to `Function` 
to avoid deleting it during graph transformations. You can do it with the constructor, as shown in the example, or with the special method `add_sinks(const SinkVector& sinks)`. Also you can delete 
sink from `ngraph::Function` after deleting the node from graph with the `delete_sink()` method.

## OpenVINO state API

 Inference Engine has the `InferRequest::QueryState` method  to get the list of states from a network and `IVariableState` interface to operate with states. Below you can find brief description of methods and the workable example of how to use this interface.  
 is below and next section contains small workable example how this interface can be used.
 
 * `std::string GetName() const`
   returns name(variable_id) of according Variable
 * `void Reset()`
   reset state to default value
 * `void SetState(Blob::Ptr newState)`
   set new value for state
 * `Blob::CPtr GetState() const`
   returns current value of state

## Example of stateful network inference

Let's take an IR from the previous section example. The example below demonstrates inference of two independent sequences of data. State should be reset between these sequences.

One infer request and one thread 
will be used in this example. Using several threads is possible if you have several independent sequences. Then each sequence can be processed in its own infer 
request. Inference of one sequence in several infer requests is not recommended. In one infer request state will be saved automatically between inferences, but 
if the first step is done in one infer request and the second in another, state should be set in new infer request manually (using `IVariableState::SetState` method).

@snippet openvino/docs/snippets/InferenceEngine_network_with_state_infer.cpp part1

You can find more powerful examples demonstrating how to work with networks with states in speech sample and demo. 
Decsriptions can be found in [Samples Overview](./Samples_Overview.md)

[state_network_example]: ./img/state_network_example.png


## LowLatency transformation

If the original framework does not have a special API for working with states, after importing the model, OpenVINO representation will not contain Assign/ReadValue layers. For example, if the original ONNX model contains RNN operations, IR will contain TensorIterator operations and the values will be obtained only after the execution of whole TensorIterator primitive, intermediate values from each iteration will not be available. To be able to work with these intermediate values of each iteration and receive them with a low latency after each infer request, a special LowLatency transformation was introduced.

LowLatency transformation changes the structure of the network containing [TensorIterator](../ops/infrastructure/TensorIterator_1.md) by adding the ability to work with state, inserting Assign/ReadValue layers as it is shown in the picture below.

![applying_low_latency_example](./img/applying_low_latency.png)

### Steps to apply LowLatency transformation

1. Get CNNNetwork. Any way is acceptable:

	* [from IR or ONNX model](Integrate_with_customer_application_new_API.md#integration-steps)
	* [from nGraph Function](../nGraph_DG/build_function.md)

2. [Reshape](ShapeInference) CNNNetwork network if necessary 
**Necessary case:** the sequence_lengths dimention of input > 1, it means TensorIterator layer will have number_iterations > 1. We should reshape the inputs of the network to set sequence_dimension exactly to 1.
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
**State naming rule:**  a name of state is a concatenation of names: original TensorIterator operation, Parameter of the body, and additional suffix "variable_" + id (0-base indexing, new indexing for each TensorIterator), for example:
```
tensor_iterator_name = "TI_name"
body_parameter_name = "param_name"

state_name = "TI_name/param_name/variable_0"
```
4. [Use state API](#openvino-state-api)

 
### Known limitations
1. Parameters are directly connected to States (ReadValues).

	Removing Parameters from `ngraph::Function` is not possible.

	![low_latency_limitation_1](./img/low_latency_limitation_1.png)

	**Current solution:** replace Parameter with Constant (freeze) with the value [0, 0, 0 … 0] via [ModelOptimizer CLI](../MO_DG/prepare_model/convert_model/Converting_Model_General.md) `--input` or `--freeze_placeholder_with_value`.

2.  Non-reshapable network.

	Value of shapes is hard-coded somewhere in the network. 

	![low_latency_limitation_2](./img/low_latency_limitation_2.png)

	**Current solution:** trim non-reshapable layers via [ModelOptimizer CLI](../MO_DG/prepare_model/convert_model/Converting_Model_General.md) `--input`, `--output` or via nGraph.

```cpp
	// nGraph example:
	auto func = cnnNetwork.getFunction();
	auto new_const = std::make_shared<ngraph::opset5::Constant>(); // type, shape, value
	for (const auto& node : func->get_ops()) {
		if (node->get_friendly_name() == "name_of_non_reshapable_const") {
			auto bad_const = std::dynamic_pointer_cast<ngraph::opset5::Constant>(node);
			ngraph::replace_node(bad_const, new_const); // replace constant
		}
	}
```
