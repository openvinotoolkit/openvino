## RNNCell <a name="RNNCell"></a> {#openvino_docs_ops_sequence_RNNCell_1}

**Versioned name**: *RNNCell-1*

**Category**: Sequence processing

**Short description**: *RNNCell* represents a single RNN cell that computes the output using the formula described in the [article](https://hackernoon.com/understanding-architecture-of-lstm-cell-from-scratch-with-code-8da40f0b71f4).

**Attributes**

* *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* *activations*

  * **Description**: activation functions for gates
  * **Range of values**: any combination of *relu*, *sigmoid*, *tanh*
  * **Type**: a list of strings
  * **Default value**: *sigmoid,tanh*
  * **Required**: *no*

* *activations_alpha, activations_beta*

  * **Description**: *activations_alpha, activations_beta* functions attributes
  * **Range of values**: a list of floating-point numbers
  * **Type**: `float[]`
  * **Default value**: None
  * **Required**: *no*

* *clip*

  * **Description**: *clip* specifies value for tensor clipping to be in *[-C, C]* before activations
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: *infinity* that means that the clipping is not applied
  * **Required**: *no*

**Inputs**

* **1**: `X` - 2D ([batch_size, input_size]) input data. Required.

* **2**: `initial_hidden_state` - 2D ([batch_size, hidden_size]) input hidden state data. Required.

**Outputs**

* **1**: `Ho` - 2D ([batch_size, hidden_size]) output hidden state.