Stateful models {#openvino_docs_IE_DG_network_state_intro}
==============================

This section describes how to work with stateful networks in OpenVINO toolkit, specifically:
* How stateful networks are represented in IR and OpenVINO
* How operations with state can be done

The section additionally provides small examples of stateful network and code to infer it.

## What is a Stateful Network?

 Several use cases require processing of data sequences. When length of a sequence is known and small enough, 
 we can process it with RNN like networks that contain a cycle inside. But in some cases, like online speech recognition of time series 
 forecasting, length of data sequence is unknown. Then data can be divided in small portions and processed step-by-step. But dependency 
 between data portions should be addressed. For that, networks save some data between inferences - state. When one dependent sequence is over,
 state should be reset to initial value and new sequence can be started.
 
 Several frameworks have special API for states in networks. For example, Keras has special option for RNNs `stateful` that turns on saving state 
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

To get a model with states ready for inference, you can convert a model from another framework to IR with Model Optimizer or create an OpenVINO Model (details can be found in [Build OpenVINO Model section](../OV_Runtime_UG/model_representation.md)). Let's represent the following graph in both forms:

![state_network_example](./img/state_network_example.png)

### Example of IR with State

The `bin` file for this graph should contain float 0 in binary form. Content of `xml` is the following.

```xml
<?xml version="1.0"?>
<net name="adder" version="11">
	<layers>
		<layer id="0" name="input" type="Parameter" version="opset1">
			<data shape="1,1" element_type="f32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="init_const" type="Const" version="opset1">
			<data element_type="f32" shape="1, 1" offset="0" size="4" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="read" type="ReadValue" version="opset6">
			<data variable_id="variable0" />
			<input>
				<port id="0" precision="FP32">
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
		<layer id="3" name="add" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
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
			<data variable_id="variable0" />
			<input>
				<port id="0" precision="FP32">
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
		<layer id="5" name="result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="2" from-port="1" to-layer="3" to-port="1" />
		<edge from-layer="3" from-port="2" to-layer="4" to-port="0" />
		<edge from-layer="3" from-port="2" to-layer="5" to-port="0" />
	</edges>
</net>
```

### Example of Creating Model OpenVINO API

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:state_network]

@endsphinxdirective

In this example, `ov::SinkVector` is used to create `ov::Model`. For network with states, except inputs and outputs,  `Assign` nodes should also point to `Model` 
to avoid deleting it during graph transformations. You can do it with the constructor, as shown in the example, or with the special method `add_sinks(const SinkVector& sinks)`. Also, you can delete 
sink from `ov::Model` after deleting the node from graph with the `delete_sink()` method.

## OpenVINO state API

 Inference Engine has the `ov::InferRequest::query_state` method  to get the list of states from a network and `ov::VariableState` class to operate with states. 
 Below you can find brief description of methods and the workable example of how to use this interface.
 
 * `std::string get_name() const`
   returns name(variable_id) of according Variable
 * `void reset()`
   reset state to default value
 * `void set_state(const Tensor& state)`
   set new value for state
 * `Tensor get_state() const`
   returns current value of state

## Example of Stateful Network Inference

Let's take an IR from the previous section example. The example below demonstrates inference of two independent sequences of data. State should be reset between these sequences.

One infer request and one thread will be used in this example. Using several threads is possible if you have several independent sequences. Then each sequence can be processed in its own infer request. Inference of one sequence in several infer requests is not recommended. In one infer request state will be saved automatically between inferences, but 
if the first step is done in one infer request and the second in another, state should be set in new infer request manually (using `ov::VariableState::set_state` method).

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:state_api_usage]

@endsphinxdirective

You can find more powerful examples demonstrating how to work with networks with states in speech sample and demo. 
Descriptions can be found in [Samples Overview](./Samples_Overview.md)


## State related Transformations

If the original framework does not have a special API for working with states, after importing the model, OpenVINO representation will not contain Assign/ReadValue layers. For example, if the original ONNX model contains RNN operations, IR will contain TensorIterator operations and the values will be obtained only after execution of the whole TensorIterator primitive. Intermediate values from each iteration will not be available. To enable you to work with these intermediate values of each iteration and receive them with a low latency after each infer request, special LowLatency, LowLatency2 and MakeStateful transformations were introduced.

### How to get TensorIterator/Loop operations from different frameworks via ModelOptimizer.
**MXNet:** *LSTM, RNN, GRU* original layers are converted to TensorIterator operation, TensorIterator body contains LSTM/RNN/GRU Cell operations.

**ONNX and frameworks supported via ONNX format:** *LSTM, RNN, GRU* original layers are converted to the GRU/RNN/LSTM Sequence operations.
*ONNX Loop* layer is converted to the OpenVINO Loop operation.

**TensorFlow:** *BlockLSTM* is converted to TensorIterator operation, TensorIterator body contains LSTM Cell operation, Peepholes, InputForget modifications are not supported.
*While* layer is converted to TensorIterator, TensorIterator body can contain any supported operations, but dynamic cases, when count of iterations cannot be calculated in shape inference (ModelOptimizer conversion) time, are not supported.

**TensorFlow2:** *While* layer is converted to Loop operation. Loop body can contain any supported operations.

**Kaldi:** Kaldi models already contain Assign/ReadValue (Memory) operations after model conversion. TensorIterator/Loop operations are not generated.

## LowLatencу2

LowLatency2 transformation changes the structure of the network containing [TensorIterator](../ops/infrastructure/TensorIterator_1.md) and [Loop](../ops/infrastructure/Loop_5.md) by adding the ability to work with the state, inserting the Assign/ReadValue layers as it is shown in the picture below.

### The differences between LowLatency and LowLatency2**:

* Unrolling of TensorIterator/Loop operations became a part of LowLatency2, not a separate transformation. After invoking the transformation, the network can be serialized and inferred without re-invoking the transformation.
* Added support for TensorIterator and Loop operations with multiple iterations inside. TensorIterator/Loop will not be unrolled in this case.
* Resolved the ‘Parameters connected directly to ReadValues’ limitation. To apply the previous version of the transformation in this case, additional manual manipulations were required, now the case is processed automatically.
#### Example of applying LowLatency2 transformation:
![applying_low_latency_2_example](./img/applying_low_latency_2.png)

After applying the transformation, ReadValue operations can receive other operations as an input, as shown in the picture above. These inputs should set the initial value for initialization of ReadValue operations. However, such initialization is not supported in the current State API implementation. Input values are ignored and the initial values for the ReadValue operations are set to zeros unless otherwise specified by the user via [State API](#openvino-state-api).

### Steps to apply LowLatency2 Transformation

1. Get [ov::Model](../OV_Runtime_UG/model_representation.md), for example:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:get_ov_model]

@endsphinxdirective
    
2. Change the number of iterations inside TensorIterator/Loop nodes in the network using the [Reshape](../OV_Runtime_UG/ShapeInference.md) feature. 

For example, the *sequence_lengths* dimension of input of the network > 1, it means the TensorIterator layer has number_of_iterations > 1. 
You can reshape the inputs of the network to set *sequence_dimension* to exactly 1.

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:reshape_ov_model]

@endsphinxdirective

**Unrolling**: If the LowLatency2 transformation is applied to a network containing TensorIterator/Loop nodes with exactly one iteration inside, these nodes are unrolled; otherwise, the nodes remain as they are. Please see [the picture](#example-of-applying-lowlatency2-transformation) for more details.

3. Apply LowLatency2 transformation

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:apply_low_latency_2]

@endsphinxdirective

**Use_const_initializer argument**

By default, the LowLatency2 transformation inserts a constant subgraph of the same shape as the previous input node, and with zero values as the initializing value for ReadValue nodes, please see the picture below. We can disable insertion of this subgraph by passing the `false` value for the `use_const_initializer` argument.

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:low_latency_2_use_parameters]

@endsphinxdirective

![use_const_initializer_example](./img/llt2_use_const_initializer.png)

**State naming rule:**  a name of a state is a concatenation of names: original TensorIterator operation, Parameter of the body, and additional suffix "variable_" + id (0-base indexing, new indexing for each TensorIterator). You can use these rules to predict what the name of the inserted State will be after the transformation is applied. For example:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:low_latency_2]

@endsphinxdirective

4. Use state API. See sections [OpenVINO state API](#openvino-state-api), [Example of stateful network inference](#example-of-stateful-network-inference).

### Known Limitations
1. Unable to execute [Reshape](ShapeInference.md) to change the number iterations of TensorIterator/Loop layers to apply the transformation correctly due to hardcoded values of shapes somewhere in the network.

	The only way you can change the number iterations of TensorIterator/Loop layer is to use the Reshape feature, but networks can be non-reshapable, the most common reason is that the value of shapes is hardcoded in a constant somewhere in the network. 

	![low_latency_limitation_2](./img/low_latency_limitation_2.png)

	**Current solution:** Trim non-reshapable layers via [ModelOptimizer CLI](../MO_DG/prepare_model/convert_model/Converting_Model.md) `--input`, `--output`. For example, the parameter and the problematic constant in the picture above can be trimmed using the following command line option: 
	`--input Reshape_layer_name`. The problematic constant can be also replaced using OpenVINO, as shown in the example below.

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:replace_const]

@endsphinxdirective

## MakeStateful

MakeStateful transformation changes the structure of the network by adding the ability to work with the state,
replacing provided by user Parameter/Results with Assign/ReadValue operations as it is shown in the picture below.

![simple_example](./img/make_stateful_simple.png)

State naming rule: in most cases, a name of a state is a concatenation of Parameter/Result tensor names. If there are no tensor names, friendly names are used.

Examples:

Detailed illustration for all examples below:
![detailed_illustration](./img/make_stateful_detailed.png)

1. C++ API

Using tensor names:

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:make_stateful_tensor_names]

@endsphinxdirective

Using Parameter/Result operations:

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:make_stateful_ov_nodes]

@endsphinxdirective

2. ModelOptimizer command line

Using tensor names:
```
--input_model <INPUT_MODEL> --transform "MakeStateful[param_res_names={'tensor_name_1':'tensor_name_4','tensor_name_3':'tensor_name_6'}]"
```

**Note:**
Only strict syntax is supported, as in the example above, the transformation call must be in double quotes 
"MakeStateful[...]", the tensor names in single quotes 'tensor_name_1' and without spaces.
