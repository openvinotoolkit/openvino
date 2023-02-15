# Stateful models {#openvino_docs_OV_UG_network_state_intro}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_lowlatency2
   openvino_docs_OV_UG_lowlatency_deprecated

@endsphinxdirective

Several use cases require processing of data sequences. When length of a sequence is known and small enough, 
it can be processed with RNN like networks that contain a cycle inside. However, in some cases (e.g., online speech recognition of time series 
forecasting) length of data sequence is unknown. Then, data can be divided in small portions and processed step-by-step. The dependency 
between data portions should be addressed. For that, networks save some data between inferences - a state. When one dependent sequence is over,
a state should be reset to initial value and a new sequence can be started.

Several frameworks have special APIs for states in networks. For example, Keras has `stateful` - a special option for RNNs, that turns on saving a state between inferences. Kaldi contains special `Offset` specifier to define time offset in a network.

OpenVINO also contains a special API to simplify work with networks with states. A state is automatically saved between inferences, 
and there is a way to reset a state when needed. A state can also be read or set to some new value between inferences.
 
## OpenVINO State Representation

OpenVINO contains the `Variable`, a special abstraction to represent a state in a network. There are two operations: [Assign](../ops/infrastructure/Assign_3.md) - to save a value in a state and [ReadValue](../ops/infrastructure/ReadValue_3.md) - to read a value saved on previous iteration.
 
To get a model with states ready for inference, convert a model from another framework to OpenVINO IR with Model Optimizer or create an nGraph function. 
(For more information, refer to the [Build OpenVINO Model section](../OV_Runtime_UG/model_representation.md)). 

Below is the graph in both forms:

@sphinxdirective

.. image:: _static/images/state_network_example.svg
   :scale: 80 %

@endsphinxdirective

### Example of IR with State

The `bin` file for this graph should contain `float 0` in binary form. The content of the `xml` file is as follows.

@sphinxdirective

.. dropdown:: Click to see the XML file.

   .. code-block:: xml
   
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

@endsphinxdirective

### Example of Creating Model nGraph API

In the following example, the `SinkVector` is used to create the `ngraph::Function`. For a network with states, except inputs and outputs, the `Assign` nodes should also point to the `Function` to avoid deleting it during graph transformations. Use the constructor to do it, as shown in the example, or with the special `add_sinks(const SinkVector& sinks)` method. After deleting the node from the graph with the `delete_sink()` method, a sink can be deleted from `ngraph::Function`.

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

@anchor openvino-state-api
## OpenVINO State API

Inference Engine has the `InferRequest::QueryState` method to get the list of states from a network and `IVariableState` interface to operate with states. Below is a brief description of methods and the example of how to use this interface.

* `std::string GetName() const` - returns the name (variable_id) of a corresponding Variable.
* `void Reset()` - resets a state to a default value.
* `void SetState(Blob::Ptr newState)` - sets a new value for a state.
* `Blob::CPtr GetState() const` - returns current value of state.

@anchor example-of-stateful-network-inference
## Example of Stateful Network Inference

Based on the IR from the previous section, the example below demonstrates inference of two independent sequences of data. A state should be reset between these sequences.

One infer request and one thread will be used in this example. Using several threads is possible if there are several independent sequences. Then, each sequence can be processed in its own infer request. Inference of one sequence in several infer requests is not recommended. In one infer request, a state will be saved automatically between inferences, but if the first step is done in one infer request and the second in another, a state should be set in a new infer request manually (using the `IVariableState::SetState` method).

@snippet openvino/docs/snippets/InferenceEngine_network_with_state_infer.cpp part1

For more elaborate examples demonstrating how to work with networks with states, refer to the speech sample and a demo in the [Samples Overview](./Samples_Overview.md). 

## LowLatency Transformations

If the original framework does not have a special API for working with states, OpenVINO representation will not contain `Assign`/`ReadValue` layers after importing the model. For example, if the original ONNX model contains RNN operations, OpenVINO IR will contain [TensorIterator](../ops/infrastructure/TensorIterator_1.md) operations and the values will be obtained only after execution of the whole `TensorIterator` primitive. Intermediate values from each iteration will not be available. Working with these intermediate values of each iteration is enabled by special [LowLatency](lowlatency_deprecated.md) and [LowLatency2](lowlatency2.md) transformations, which also help receive these values with a low latency after each infer request.

> **NOTE**: It is recommended to use LowLatency2, as LowLatency transformation has already been deprecated.

### TensorIterator/Loop operations

You can get the TensorIterator/Loop operations from different frameworks via Model Optimizer.

* **ONNX and frameworks supported via ONNX format** - `LSTM`, `RNN`, and `GRU` original layers are converted to the `TensorIterator` operation. The `TensorIterator` 
body contains `LSTM`/`RNN`/`GRU Cell`. The `Peepholes` and `InputForget` modifications are not supported, while the `sequence_lengths` optional input is.
`ONNX Loop` layer is converted to the OpenVINO [Loop](../ops/infrastructure/Loop_5.md) operation.

* **Apache MXNet** - `LSTM`, `RNN`, `GRU` original layers are converted to `TensorIterator` operation, which body contains `LSTM`/`RNN`/`GRU Cell` operations.

* **TensorFlow** - `BlockLSTM` is converted to `TensorIterator` operation. The `TensorIterator` body contains `LSTM Cell` operation, whereas `Peepholes` and `InputForget` modifications are not supported.
The `While` layer is converted to `TensorIterator`, which body can contain any supported operations. However, when count of iterations cannot be calculated in shape inference (Model Optimizer conversion) time, the dynamic cases are not supported.

* **TensorFlow2** - `While` layer is converted to `Loop` operation, which body can contain any supported operations.

* **Kaldi** - Kaldi models already contain `Assign`/`ReadValue` (Memory) operations after model conversion. The `TensorIterator`/`Loop` operations are not generated.
