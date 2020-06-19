## LSTMSequence <a name="LSTMSequence"></a>

**Versioned name**: *LSTMSequence-1*

**Category**: *Sequence processing*

**Short description**: *LSTMSequence* operation represents a series of LSTM cells. Each cell is implemented as <a href="#LSTMCell">LSTMCell</a> operation.

**Detailed description**

A single cell in the sequence is implemented in the same way as in <a href="#LSTMCell">LSTMCell</a> operation. *LSTMSequence* represents a sequence of LSTM cells. The sequence can be connected differently depending on `direction` attribute that specifies the direction of traversing of input data along sequence dimension or specifies whether it should be a bidirectional sequence. The most of the attributes are in sync with the specification of ONNX LSTM operator defined <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#lstm">LSTMCell</a>.


**Attributes**

* *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* *activations*

  * **Description**: *activations* specifies activation functions for gates, there are three gates, so three activation functions should be specified as a value for this attributes
  * **Range of values**: any combination of *relu*, *sigmoid*, *tanh*
  * **Type**: a list of strings
  * **Default value**: *sigmoid,tanh,tanh*
  * **Required**: *no*

* *activations_alpha, activations_beta*

  * **Description**: *activations_alpha, activations_beta* attributes of functions; applicability and meaning of these attributes depends on choosen activation functions
  * **Range of values**: a list of floating-point numbers
  * **Type**: `float[]`
  * **Default value**: None
  * **Required**: *no*

* *clip*

  * **Description**: *clip* specifies bound values *[-C, C]* for tensor clipping. Clipping is performed before activations.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: *infinity* that means that the clipping is not applied
  * **Required**: *no*

* *direction*

  * **Description**: Specify if the RNN is forward, reverse, or bidirectional. If it is one of *forward* or *reverse* then `num_directions = 1`, if it is *bidirectional*, then `num_directions = 2`. This `num_directions` value specifies input/output shape requirements.
  * **Range of values**: *forward*, *reverse*, *bidirectional*
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *Yes*

**Inputs**

* **1**: `X` - 3D ([batch_size, seq_length, input_size]) input data. It differs from LSTMCell 1st input only by additional axis with size `seq_length`. Floating point type. Required.

* **2**: `initial_hidden_state` - 3D ([batch_size, num_directions, hidden_size]) input hidden state data. Floating point type. Required.

* **3**: `initial_cell_state` - 3D ([batch_size, num_directions, hidden_size]) input cell state data. Floating point type. Required.

* **4**: `sequence_lengths` - 1D ([batch_size]) specifies real sequence lengths for each batch element. Integer type. Required.

* **5**: `W` - 3D tensor with weights for matrix multiplication operation with input portion of data, shape is `[num_directions, 4 * hidden_size, input_size]`, output gate order: fico. Floating point type. Required.

* **6**: `R` - 3D tensor with weights for matrix multiplication operation with hidden state, shape is `[num_directions, 4 * hidden_size, hidden_size]`, output gate order: fico. Floating point type. Required.

* **7**: `B` - 2D tensor with biases, shape is `[num_directions, 4 * hidden_size]`. Floating point type. Required.

**Outputs**

* **1**: `Y` – 3D output, shape [batch_size, num_directions, seq_len, hidden_size]

* **2**: `Ho` - 3D ([batch_size, num_directions, hidden_size]) output hidden state.

* **3**: `Co` - 3D ([batch_size, num_directions, hidden_size]) output cell state.