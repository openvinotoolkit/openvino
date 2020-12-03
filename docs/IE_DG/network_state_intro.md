Introduction to OpenVINO state API {#openvino_docs_IE_DG_network_state_intro}
==============================

This section provides description how to work with stateful networks in OpenVINO: how such networks 
are represented in IR and Ngraph, how operations with state can be done. Section provides small examples 
of stateful network and code to infer it.

## What is stateful network

 Several use cases require processing of data sequences. When length of sequence is known and small enough, 
 we can process it with RNN like networks which contains cycle inside. But in some cases like online speech recognition of time series 
 forecasting length of data sequence is unknown. Then data can be divided in small portions and processed step by step. But dependency 
 between data portions should be addressed. For that networks saved some data between inferences - state. When one dependent sequence is over,
 state should be reset to initial value and new sequence can be started.
 
 Several frameworks have special API for states in networks. For example, Keras have special option for RNNs `stateful` that turn on saving state 
 between inferences. Kaldi contains special specifier `Offset` to define time offset in network. 
 
 OpenVINO contains special API to simplify work with networks with states too. State will be automatically saved between inferences 
 and there is a way to reset state when needed. Also user can read state or set it to some new value between inferences.
 
## OpenVINO state representation

 OpenVINO contains special abstraction Variable to represent state in network. There are 2 operations to work with state: 
* `Assign` to save value in state
* `ReadValue` to read value saved on previous iteration.

More details on these operations you can find in [ReadValue specification](../ops/infrastructure/ReadValue_3.md) and 
[Assign specification](../ops/infrastructure/Assign_3.md).

## Examples of representation of network with states

To get model with states ready for inference you can convert model from another framework with ModelOptimizer to IR or create ngraph function 
(details can be found in [Build nGraph Function section](../nGraph_DG/build_function.md)). 
Let's represent in both forms the following graph:
![state_network_example]

### Example of IR with state

`bin` file for this graph should contain float 0 in binary form. `xml` is the following.

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

### Example of creating model Ngraph API

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

In this example `SinkVector` was used to create `ngraph::Function`. For network with states except inputs and outputs also Assign nodes should be pointed to Function 
to avoid deleting it during graph transformations. It can be done with constructor as shown in example or with special method `Function::add_sink`. Also you can delete 
sink from `Function` after deleting node from graph with `Function::delete_sink` method.

## OpenVINO state API

 InferenceEngine have method `InferRequest::QueryState` to get list of states from network and IVariableState interface to operate with state. Brief description of methods 
 is below and next section contains small workable example how this interface can be used.
 
 * `std::string GetName() const`
   returns name(variable_id) of according Variable
 * `void Reset()`
   reset state to default value
 * `void SetState(Blob::Ptr newState)`
   set new value for state
 * `Blob::CPtr GetState() const`
   returns current value of state

## Example of inference stateful network

Let's take IR from example in prevous sections. In this example inference of 2 independent sequences of data will be demonstrated. Between these sequences state should be reset.

One infer request and one thread 
will be used in this example. Using several threads is possible if you have several independent sequences. Then each sequence can be processed in its own infer 
request. Inference of one sequence in several infer requests is not recommended. In one infer request state will be saved automatically between inferences, but 
if the first step done in one infer request and the second in another, state should be set in new infer request manually (using `IVariableState::SetState` method).

@snippet openvino/docs/snippets/InferenceEngine_network_with_state_infer.cpp part1

More powerfull examples of work with networks with states are sample and demo demonstrating work with speech. 
Decsriptions can be found in [Samples Overview](./Samples_Overview.md)

[state_network_example]: ./img/state_network_example.png