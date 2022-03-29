Ways to get stateful models in OpenVINO  {#openvino_docs_OV_UG_ways_to_get_stateful_model}
==============================

## State related Transformations

If the original framework does not have a special API for working with states, after importing the model, OpenVINO representation will not contain Assign/ReadValue layers.
For example, if the original ONNX model contains RNN operations, IR will contain TensorIterator operations and the values will be obtained only after execution of the whole TensorIterator primitive.
Intermediate values from each iteration will not be available. To enable you to work with these intermediate values of each iteration and receive them with a low latency after each infer request,
special LowLatency2 and MakeStateful transformations were introduced.

### How to get TensorIterator/Loop operations from different frameworks via ModelOptimizer.
**MXNet:** *LSTM, RNN, GRU* original layers are converted to TensorIterator operation, TensorIterator body contains LSTM/RNN/GRU Cell operations.

**ONNX and frameworks supported via ONNX format:** *LSTM, RNN, GRU* original layers are converted to the GRU/RNN/LSTM Sequence operations.
*ONNX Loop* layer is converted to the OpenVINO Loop operation.

**TensorFlow:** *BlockLSTM* is converted to TensorIterator operation, TensorIterator body contains LSTM Cell operation, Peepholes, InputForget modifications are not supported.
*While* layer is converted to TensorIterator, TensorIterator body can contain any supported operations, but dynamic cases, when count of iterations cannot be calculated in shape inference (ModelOptimizer conversion) time, are not supported.

**TensorFlow2:** *While* layer is converted to Loop operation. Loop body can contain any supported operations.

**Kaldi:** Kaldi models already contain Assign/ReadValue (Memory) operations after model conversion. TensorIterator/Loop operations are not generated.

## LowLatencу2

LowLatency2 transformation changes the structure of the network containing [TensorIterator](../ops/infrastructure/TensorIterator_1.md) 
and [Loop](../ops/infrastructure/Loop_5.md) by adding the ability to work with the state, inserting the Assign/ReadValue 
layers as it is shown in the picture below.

#### Example of applying LowLatency2 transformation:
![applying_low_latency_2_example](./img/applying_low_latency_2.png)

After applying the transformation, ReadValue operations can receive other operations as an input, as shown in the picture above. 
These inputs should set the initial value for initialization of ReadValue operations. 
However, such initialization is not supported in the current State API implementation. 
Input values are ignored and the initial values for the ReadValue operations are set to zeros unless otherwise specified 
by the user via [State API](#openvino-state-api).

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

State naming rule: in most cases, a name of a state is a concatenation of Parameter/Result tensor names. 
If there are no tensor names, [friendly names](../Extensibility_UG/ov_transformations.md#1-friendly-names) are used.

Examples:

Detailed illustration for all examples below:
![detailed_illustration](./img/make_stateful_detailed.png)

1. C++ API

Using tensor names:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:make_stateful_tensor_names]

@endsphinxdirective

Using Parameter/Result operations:

@sphinxdirective

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

## How to create a network with state using OpenVINO

To get a model with states ready for inference, you can convert a model from another framework to IR with Model Optimizer 
or create an OpenVINO Model (details can be found in [Build OpenVINO Model section](../OV_Runtime_UG/model_representation.md)).
Let's build the following graph using C++ OpenVINO API:

![state_network_example](./img/state_network_example.png)

### Example of Creating Model via OpenVINO API

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:state_network]

@endsphinxdirective

In this example, `ov::SinkVector` is used to create `ov::Model`. For network with states, except inputs and outputs,  `Assign` nodes should also point to `Model` 
to avoid deleting it during graph transformations. You can do it with the constructor, as shown in the example, or with the special method `add_sinks(const SinkVector& sinks)`. Also, you can delete 
sink from `ov::Model` after deleting the node from graph with the `delete_sink()` method.
This model 
